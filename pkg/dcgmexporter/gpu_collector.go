/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dcgmexporter

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"github.com/sirupsen/logrus"
)

type DCGMCollectorConstructor func([]Counter, string, *Config, FieldEntityGroupTypeSystemInfoItem) (*DCGMCollector, func(), error)

func NewDCGMCollector(c []Counter,
	hostname string,
	config *Config,
	fieldEntityGroupTypeSystemInfo FieldEntityGroupTypeSystemInfoItem) (*DCGMCollector, func(), error) {

	if fieldEntityGroupTypeSystemInfo.isEmpty() {
		return nil, func() {}, errors.New("fieldEntityGroupTypeSystemInfo is empty")
	}

	collector := &DCGMCollector{
		Counters:     c,
		DeviceFields: fieldEntityGroupTypeSystemInfo.DeviceFields,
		SysInfo:      fieldEntityGroupTypeSystemInfo.SystemInfo,
		Hostname:     hostname,
	}

	if config == nil {
		logrus.Warn("Config is empty")
		return collector, func() { collector.Cleanup() }, nil
	}

	collector.UseOldNamespace = config.UseOldNamespace
	collector.ReplaceBlanksInModelName = config.ReplaceBlanksInModelName

	cleanups, err := SetupDcgmFieldsWatch(collector.DeviceFields,
		fieldEntityGroupTypeSystemInfo.SystemInfo,
		int64(config.CollectInterval)*1000)
	if err != nil {
		logrus.Fatal("Failed to watch metrics: ", err)
	}

	collector.Cleanups = cleanups

	return collector, func() { collector.Cleanup() }, nil
}

func GetSystemInfo(config *Config, entityType dcgm.Field_Entity_Group) (*SystemInfo, error) {
	sysInfo, err := InitializeSystemInfo(config.GPUDevices,
		config.SwitchDevices,
		config.CPUDevices,
		config.UseFakeGPUs, entityType)
	if err != nil {
		return nil, err
	}
	return &sysInfo, err
}

func GetHostname(config *Config) (string, error) {
	hostname := ""
	var err error
	if !config.NoHostname {
		if nodeName := os.Getenv("NODE_NAME"); nodeName != "" {
			hostname = nodeName
		} else {
			hostname, err = os.Hostname()
			if err != nil {
				return "", err
			}
		}
	}
	return hostname, nil
}

func (c *DCGMCollector) Cleanup() {
	for _, c := range c.Cleanups {
		c()
	}
}

func generateMigCache(monitoringInfo []MonitoringInfo) map[uint]*MigResources {
	migResourceCache := make(map[uint]*MigResources)
	for _, mi := range monitoringInfo {
		var vals []dcgm.FieldValue_v1
		var err error
		fileds := []dcgm.Short{dcgm.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, dcgm.DCGM_FI_PROF_DRAM_ACTIVE, dcgm.DCGM_FI_PROF_PIPE_FP64_ACTIVE, dcgm.DCGM_FI_PROF_PIPE_FP32_ACTIVE, dcgm.DCGM_FI_PROF_PIPE_FP16_ACTIVE}
		// Added else for testsing in non mig system
		if mi.InstanceInfo != nil {
			vals, err = dcgm.EntityGetLatestValues(mi.Entity.EntityGroupId, mi.Entity.EntityId, fileds)
		} else {
			return nil
		}
		if err != nil {
			if derr, ok := err.(*dcgm.DcgmError); ok {
				if derr.Code == dcgm.DCGM_ST_CONNECTION_NOT_VALID {
					logrus.Fatal("Could not retrieve metrics: ", err)
				}
			}
		}
		var migCache *MigResources
		migCache, ok := migResourceCache[mi.DeviceInfo.GPU]
		if !ok {
			migCache = &MigResources{0.0, 0.0, 0.0, 0.0, 0.0}
		}
		for _, val := range vals {
			v := ToString(val)
			if v == SkipDCGMValue {
				continue
			}
			float_value, err := strconv.ParseFloat(v, 64)
			if err != nil {
				continue
			}
			if val.FieldId == 1004 {
				migCache.Tensor += float_value
			} else if val.FieldId == 1005 {
				migCache.Dram += float_value
			} else if val.FieldId == 1006 {
				migCache.FP64 += float_value
			} else if val.FieldId == 1007 {
				migCache.FP32 += float_value
			} else if val.FieldId == 1008 {
				migCache.FP16 += float_value
			} else {
				continue
			}
		}

		v, ok := migResourceCache[mi.DeviceInfo.GPU]
		if !ok {
			migResourceCache[mi.DeviceInfo.GPU] = migCache
		}
	}
	fmt.Printf("\nMig resource cache : %+v\n", migResourceCache)
	return migResourceCache
}

func (c *DCGMCollector) GetMetrics() (MetricsByCounter, error) {
	monitoringInfo := GetMonitoredEntities(c.SysInfo)

	migResourceCache := generateMigCache(monitoringInfo)

	metrics := make(MetricsByCounter)

	for _, mi := range monitoringInfo {
		var vals []dcgm.FieldValue_v1
		var err error
		if mi.Entity.EntityGroupId == dcgm.FE_LINK {
			vals, err = dcgm.LinkGetLatestValues(mi.Entity.EntityId, mi.ParentId, c.DeviceFields)
		} else {
			vals, err = dcgm.EntityGetLatestValues(mi.Entity.EntityGroupId, mi.Entity.EntityId, c.DeviceFields)
		}

		if err != nil {
			if derr, ok := err.(*dcgm.DcgmError); ok {
				if derr.Code == dcgm.DCGM_ST_CONNECTION_NOT_VALID {
					logrus.Fatal("Could not retrieve metrics: ", err)
				}
			}
			return nil, err
		}

		// InstanceInfo will be nil for GPUs
		if c.SysInfo.InfoType == dcgm.FE_SWITCH || c.SysInfo.InfoType == dcgm.FE_LINK {
			ToSwitchMetric(metrics, vals, c.Counters, mi, c.UseOldNamespace, c.Hostname)
		} else if c.SysInfo.InfoType == dcgm.FE_CPU || c.SysInfo.InfoType == dcgm.FE_CPU_CORE {
			ToCPUMetric(metrics, vals, c.Counters, mi, c.UseOldNamespace, c.Hostname)
		} else {
			ToMetric(metrics,
				vals,
				c.Counters,
				mi.DeviceInfo,
				mi.InstanceInfo,
				c.UseOldNamespace,
				c.Hostname,
				c.ReplaceBlanksInModelName,
				migResourceCache)
		}
	}

	return metrics, nil
}

func ShouldMonitorDeviceType(fields []dcgm.Short, entityType dcgm.Field_Entity_Group) bool {
	if len(fields) == 0 {
		return false
	}

	if len(fields) == 1 && fields[0] == dcgm.DCGM_FI_DRIVER_VERSION {
		return false
	}

	return true
}

func FindCounterField(c []Counter, fieldId uint) (Counter, error) {
	for i := 0; i < len(c); i++ {
		if uint(c[i].FieldID) == fieldId {
			return c[i], nil
		}
	}

	return c[0], fmt.Errorf("could not find counter corresponding to field ID '%d'", fieldId)
}

func ToSwitchMetric(metrics MetricsByCounter,
	values []dcgm.FieldValue_v1, c []Counter, mi MonitoringInfo, useOld bool, hostname string) {
	labels := map[string]string{}

	for _, val := range values {
		v := ToString(val)
		// Filter out counters with no value and ignored fields for this entity

		counter, err := FindCounterField(c, val.FieldId)
		if err != nil {
			continue
		}

		if counter.PromType == "label" {
			labels[counter.FieldName] = v
			continue
		}
		uuid := "UUID"
		if useOld {
			uuid = "uuid"
		}
		var m Metric
		if v == SkipDCGMValue {
			continue
		} else {
			m = Metric{
				Counter:      counter,
				Value:        v,
				UUID:         uuid,
				GPU:          fmt.Sprintf("%d", mi.Entity.EntityId),
				GPUUUID:      "",
				GPUDevice:    fmt.Sprintf("nvswitch%d", mi.ParentId),
				GPUModelName: "",
				Hostname:     hostname,
				Labels:       labels,
				Attributes:   nil,
			}
		}

		metrics[m.Counter] = append(metrics[m.Counter], m)
	}
}

func ToCPUMetric(metrics MetricsByCounter,
	values []dcgm.FieldValue_v1, c []Counter, mi MonitoringInfo, useOld bool, hostname string) {
	var labels = map[string]string{}

	for _, val := range values {
		v := ToString(val)
		// Filter out counters with no value and ignored fields for this entity

		counter, err := FindCounterField(c, val.FieldId)
		if err != nil {
			continue
		}

		if counter.PromType == "label" {
			labels[counter.FieldName] = v
			continue
		}
		uuid := "UUID"
		if useOld {
			uuid = "uuid"
		}
		var m Metric
		if v == SkipDCGMValue {
			continue
		} else {
			m = Metric{
				Counter:      counter,
				Value:        v,
				UUID:         uuid,
				GPU:          fmt.Sprintf("%d", mi.Entity.EntityId),
				GPUUUID:      "",
				GPUDevice:    fmt.Sprintf("%d", mi.ParentId),
				GPUModelName: "",
				Hostname:     hostname,
				Labels:       labels,
				Attributes:   nil,
			}
		}

		metrics[m.Counter] = append(metrics[m.Counter], m)
	}
}

func migDeviceResource(v, profile, uuid string, gpu uint, counter Counter, migResourceCache map[uint]*MigResources) string {
	if counter.FieldID != 155 {
		return v
	}
	fmt.Printf("\nAttributing mig resource metric %+v\nCurrent value %s, Profile %s\n", counter, v, profile)
	scaling_factor, err := strconv.Atoi(string(profile[0]))
	if err != nil {
		fmt.Println(err)
		return v
	}
	value, err := strconv.ParseFloat(v, 64)
	if err != nil {
		fmt.Println(err)
		return v
	}

	// Divide Idle power (Divide by scaling factor)
	// How to get Idle power (Take minimum?)
	scaled_idle_power := 90.0 * float64(scaling_factor) / 7

	// Divide Active Power
	active_power := value - 90.0
	// TODO
	scaled_active_power := active_power * float64(scaling_factor) / 7
	total_power := scaled_active_power + scaled_idle_power
	fmt.Printf("\tScaled value %f\n", total_power)
	return fmt.Sprintf("%f", total_power)
}

func ToMetric(
	metrics MetricsByCounter,
	values []dcgm.FieldValue_v1,
	c []Counter,
	d dcgm.Device,
	instanceInfo *GPUInstanceInfo,
	useOld bool,
	hostname string,
	replaceBlanksInModelName bool,
	migResourceCache map[uint]*MigResources,
) {
	var labels = map[string]string{}

	for _, val := range values {
		v := ToString(val)
		// Filter out counters with no value and ignored fields for this entity
		if v == SkipDCGMValue {
			continue
		}

		counter, err := FindCounterField(c, val.FieldId)
		if err != nil {
			continue
		}

		if counter.PromType == "label" {
			labels[counter.FieldName] = v
			continue
		}
		uuid := "UUID"
		if useOld {
			uuid = "uuid"
		}

		gpuModel := getGPUModel(d, replaceBlanksInModelName)

		m := Metric{
			Counter: counter,
			Value:   v,

			UUID:         uuid,
			GPU:          fmt.Sprintf("%d", d.GPU),
			GPUUUID:      d.UUID,
			GPUDevice:    fmt.Sprintf("nvidia%d", d.GPU),
			GPUModelName: gpuModel,
			Hostname:     hostname,

			Labels:     labels,
			Attributes: map[string]string{},
		}
		if instanceInfo != nil {
			m.MigProfile = instanceInfo.ProfileName
			m.GPUInstanceID = fmt.Sprintf("%d", instanceInfo.Info.NvmlInstanceId)
			m.Value = migDeviceResource(v, instanceInfo.ProfileName, d.UUID, d.GPU, counter, migResourceCache)
		} else {
			m.MigProfile = ""
			m.GPUInstanceID = ""
		}

		metrics[m.Counter] = append(metrics[m.Counter], m)
	}
}

func getGPUModel(d dcgm.Device, replaceBlanksInModelName bool) string {
	gpuModel := d.Identifiers.Model

	if replaceBlanksInModelName {
		parts := strings.Fields(gpuModel)
		gpuModel = strings.Join(parts, " ")
		gpuModel = strings.ReplaceAll(gpuModel, " ", "-")
	}
	return gpuModel
}

func ToString(value dcgm.FieldValue_v1) string {
	switch value.FieldType {
	case dcgm.DCGM_FT_INT64:
		switch v := value.Int64(); v {
		case dcgm.DCGM_FT_INT32_BLANK:
			return SkipDCGMValue
		case dcgm.DCGM_FT_INT32_NOT_FOUND:
			return SkipDCGMValue
		case dcgm.DCGM_FT_INT32_NOT_SUPPORTED:
			return SkipDCGMValue
		case dcgm.DCGM_FT_INT32_NOT_PERMISSIONED:
			return SkipDCGMValue
		case dcgm.DCGM_FT_INT64_BLANK:
			return SkipDCGMValue
		case dcgm.DCGM_FT_INT64_NOT_FOUND:
			return SkipDCGMValue
		case dcgm.DCGM_FT_INT64_NOT_SUPPORTED:
			return SkipDCGMValue
		case dcgm.DCGM_FT_INT64_NOT_PERMISSIONED:
			return SkipDCGMValue
		default:
			return fmt.Sprintf("%d", value.Int64())
		}
	case dcgm.DCGM_FT_DOUBLE:
		switch v := value.Float64(); v {
		case dcgm.DCGM_FT_FP64_BLANK:
			return SkipDCGMValue
		case dcgm.DCGM_FT_FP64_NOT_FOUND:
			return SkipDCGMValue
		case dcgm.DCGM_FT_FP64_NOT_SUPPORTED:
			return SkipDCGMValue
		case dcgm.DCGM_FT_FP64_NOT_PERMISSIONED:
			return SkipDCGMValue
		default:
			return fmt.Sprintf("%f", value.Float64())
		}
	case dcgm.DCGM_FT_STRING:
		switch v := value.String(); v {
		case dcgm.DCGM_FT_STR_BLANK:
			return SkipDCGMValue
		case dcgm.DCGM_FT_STR_NOT_FOUND:
			return SkipDCGMValue
		case dcgm.DCGM_FT_STR_NOT_SUPPORTED:
			return SkipDCGMValue
		case dcgm.DCGM_FT_STR_NOT_PERMISSIONED:
			return SkipDCGMValue
		default:
			return v
		}
	}

	return FailedToConvert
}
