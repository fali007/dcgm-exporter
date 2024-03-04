package dcgmexporter

import (
	"errors"
	"fmt"
	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"github.com/sirupsen/logrus"
	"os"
	"strconv"
	"strings"
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
func generateMigCache(monitoringInfo []MonitoringInfo) map[uint][]MigResources {
	migResourceCache := make(map[uint][]MigResources)
	for _, mi := range monitoringInfo {
		var vals []dcgm.FieldValue_v1
		var err error
		fileds := []dcgm.Short{dcgm.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, dcgm.DCGM_FI_PROF_DRAM_ACTIVE, dcgm.DCGM_FI_PROF_PIPE_FP64_ACTIVE, dcgm.DCGM_FI_PROF_PIPE_FP32_ACTIVE, dcgm.DCGM_FI_PROF_PIPE_FP16_ACTIVE}

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
		migCache := MigResources{}
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
				migCache.ResourceCache.Tensor = float_value
			} else if val.FieldId == 1005 {
				migCache.ResourceCache.Dram = float_value
			} else if val.FieldId == 1006 {
				migCache.ResourceCache.FP64 = float_value
			} else if val.FieldId == 1007 {
				migCache.ResourceCache.FP32 = float_value
			} else if val.FieldId == 1008 {
				migCache.ResourceCache.FP16 = float_value
			} else {
				continue
			}
		}

		migCache.Profile = mi.InstanceInfo.ProfileName
		migCache.ID = fmt.Sprintf("%d", mi.InstanceInfo.Info.NvmlInstanceId)

		v, ok := migResourceCache[mi.DeviceInfo.GPU]
		if ok {
			migResourceCache[mi.DeviceInfo.GPU] = append(v, migCache)
		} else {
			migResourceCache[mi.DeviceInfo.GPU] = []MigResources{migCache}
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
func min(a, b float64) float64 {
	if a > b {
		return b
	}
	return a
}
func migDeviceResource(v, profile, id string, gpu uint, counter Counter, migResourceCache map[uint][]MigResources) string {
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
	scaled_idle_power := min(90.0, value) * float64(scaling_factor) / 7

	// Divide Active Power
	active_power := value - min(90.0, value)
	// TODO
	// Missing part - scaling based on mig size
	cachedResource, ok := migResourceCache[gpu]
	if !ok {
		return v
	}
	scaled_active_power, err := processMigCacheForPower(cachedResource, id, active_power)
	if err != nil {
		scaled_active_power = active_power * float64(scaling_factor) / 7
	}
	total_power := scaled_active_power + scaled_idle_power
	fmt.Printf("\tScaled value %f\n", total_power)
	return fmt.Sprintf("%f", total_power)
}
func processMigCacheForPower(m []MigResources, id string, active_power float64) (float64, error) {
	totalResource := MigResourceCache{}
	var mig_instance MigResources
	var feature_weights MigResourceCache = MigResourceCache{0.338, 0.152, 0.17, 0.17, 0.17}

	for _, device := range m {
		// Scale wrt mig profile and weights
		s_factor, err := strconv.Atoi(string(device.Profile[0]))
		if err != nil {
			fmt.Println("No profile scaling factor found")
			return 0.0, errors.New("No profile scaling factor found")
		}
		scaling_factor := float64(s_factor)
		device.ResourceCache.Tensor = device.ResourceCache.Tensor * scaling_factor * feature_weights.Tensor
		totalResource.Tensor += device.ResourceCache.Tensor
		device.ResourceCache.Dram = device.ResourceCache.Dram * scaling_factor * feature_weights.Dram
		totalResource.Dram += device.ResourceCache.Dram
		device.ResourceCache.FP64 = device.ResourceCache.FP64 * scaling_factor * feature_weights.FP64
		totalResource.FP64 += device.ResourceCache.FP64
		device.ResourceCache.FP32 = device.ResourceCache.FP32 * scaling_factor * feature_weights.FP32
		totalResource.FP32 += device.ResourceCache.FP32
		device.ResourceCache.FP16 = device.ResourceCache.FP16 * scaling_factor * feature_weights.FP16
		totalResource.FP16 += device.ResourceCache.FP16
		if device.ID == id {
			mig_instance = device
		}
	}

	summed_total_metrics := totalResource.Tensor + totalResource.Dram + totalResource.FP64 + totalResource.FP32 + totalResource.FP16
	summed_instance_metrics := mig_instance.ResourceCache.Tensor + mig_instance.ResourceCache.Dram + mig_instance.ResourceCache.FP64 + mig_instance.ResourceCache.FP32 + mig_instance.ResourceCache.FP16

	active_power_scaled := active_power * summed_instance_metrics / summed_total_metrics
	fmt.Printf("Total Resource :\n%+v\nMig Instance :\n%+v\n", totalResource, mig_instance)
	if active_power_scaled != active_power_scaled {
		return 0.0, errors.New("Error computing active power")
	}
	return active_power_scaled, nil
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
	migResourceCache map[uint][]MigResources,
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
			Counter:      counter,
			Value:        v,
			UUID:         uuid,
			GPU:          fmt.Sprintf("%d", d.GPU),
			GPUUUID:      d.UUID,
			GPUDevice:    fmt.Sprintf("nvidia%d", d.GPU),
			GPUModelName: gpuModel,
			Hostname:     hostname,
			Labels:       labels,
			Attributes:   map[string]string{},
		}
		if instanceInfo != nil {
			m.MigProfile = instanceInfo.ProfileName
			m.GPUInstanceID = fmt.Sprintf("%d", instanceInfo.Info.NvmlInstanceId)
			m.Value = migDeviceResource(v, instanceInfo.ProfileName, m.GPUInstanceID, d.GPU, counter, migResourceCache)
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
