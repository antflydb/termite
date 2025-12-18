// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package controllers implements the Kubernetes controllers for Termite CRDs.
package controllers

//go:generate go tool controller-gen rbac:roleName=termite-operator-cluster-role paths="." output:rbac:artifacts:config=../manifests/rbac

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	antflyaiv1alpha1 "github.com/antflydb/termite/pkg/operator/api/v1alpha1"
)

const (
	// TermiteAPIPort is the port the Termite API server listens on.
	// This must match TERMITE_API_URL in the container image (default: http://0.0.0.0:8080).
	TermiteAPIPort = 8080
)

// TermitePoolReconciler reconciles a TermitePool object
type TermitePoolReconciler struct {
	client.Client
	Scheme       *runtime.Scheme
	TermiteImage string
}

// +kubebuilder:rbac:groups=antfly.io,resources=termitepools,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=antfly.io,resources=termitepools/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=antfly.io,resources=termitepools/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=policy,resources=poddisruptionbudgets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete

// Reconcile handles TermitePool reconciliation
func (r *TermitePoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Fetch the TermitePool
	pool := &antflyaiv1alpha1.TermitePool{}
	if err := r.Get(ctx, req.NamespacedName, pool); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("TermitePool not found, ignoring")
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	logger.Info("Reconciling TermitePool", "name", pool.Name)

	// 0. Validate configuration (fallback when webhook is disabled)
	if err := r.validatePool(pool); err != nil {
		logger.Error(err, "TermitePool validation failed")
		// Update status to reflect validation error
		pool.Status.Phase = antflyaiv1alpha1.TermitePoolPhaseDegraded
		if updateErr := r.Status().Update(ctx, pool); updateErr != nil {
			logger.Error(updateErr, "Failed to update status after validation error")
		}
		// Requeue with backoff
		return ctrl.Result{RequeueAfter: 30 * 1e9}, nil
	}

	// 1. Create or update the headless Service
	if err := r.reconcileService(ctx, pool); err != nil {
		return ctrl.Result{}, err
	}

	// 2. Create or update the ConfigMap for model configuration
	if err := r.reconcileConfigMap(ctx, pool); err != nil {
		return ctrl.Result{}, err
	}

	// 3. Create or update the StatefulSet
	if err := r.reconcileStatefulSet(ctx, pool); err != nil {
		return ctrl.Result{}, err
	}

	// 4. Create or update PodDisruptionBudget (from Availability config or GKE config)
	if err := r.reconcilePDB(ctx, pool); err != nil {
		return ctrl.Result{}, err
	}

	// 5. Update status
	if err := r.updateStatus(ctx, pool); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: 30 * 1e9}, nil // Requeue after 30 seconds
}

func (r *TermitePoolReconciler) reconcileService(ctx context.Context, pool *antflyaiv1alpha1.TermitePool) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pool.Name,
			Namespace: pool.Namespace,
			Labels:    r.labels(pool),
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone, // Headless service
			Selector:  r.selectorLabels(pool),
			Ports: []corev1.ServicePort{
				{
					Name:     "http",
					Port:     TermiteAPIPort,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}

	// Set owner reference
	if err := ctrl.SetControllerReference(pool, svc, r.Scheme); err != nil {
		return err
	}

	// Create or update
	existing := &corev1.Service{}
	if err := r.Get(ctx, types.NamespacedName{Name: svc.Name, Namespace: svc.Namespace}, existing); err != nil {
		if errors.IsNotFound(err) {
			return r.Create(ctx, svc)
		}
		return err
	}

	// Update if needed
	existing.Spec.Ports = svc.Spec.Ports
	return r.Update(ctx, existing)
}

func (r *TermitePoolReconciler) reconcileConfigMap(ctx context.Context, pool *antflyaiv1alpha1.TermitePool) error {
	// Generate complete configuration
	completeConfig, err := r.generateCompleteConfig(pool)
	if err != nil {
		return fmt.Errorf("failed to generate complete config: %w", err)
	}

	// Build model list for environment variables (backward compatibility)
	models := make([]string, 0, len(pool.Spec.Models.Preload))
	for _, m := range pool.Spec.Models.Preload {
		name := m.Name
		if m.Variant != "" {
			name = name + ":" + m.Variant
		}
		models = append(models, name)
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pool.Name + "-config",
			Namespace: pool.Namespace,
			Labels:    r.labels(pool),
		},
		Data: map[string]string{
			// Config file for --config flag
			"config.json": completeConfig,
			// Environment variables (backward compatibility)
			"TERMITE_MODELS":           strings.Join(models, ","),
			"TERMITE_POOL":             pool.Name,
			"TERMITE_WORKLOAD_TYPE":    string(pool.Spec.WorkloadType),
			"TERMITE_LOADING_STRATEGY": string(pool.Spec.Models.LoadingStrategy),
		},
	}

	if pool.Spec.Models.RegistryURL != "" {
		cm.Data["ANTFLY_REGISTRY_URL"] = pool.Spec.Models.RegistryURL
	}

	// Set owner reference
	if err := ctrl.SetControllerReference(pool, cm, r.Scheme); err != nil {
		return err
	}

	// Create or update
	existing := &corev1.ConfigMap{}
	if err := r.Get(ctx, types.NamespacedName{Name: cm.Name, Namespace: cm.Namespace}, existing); err != nil {
		if errors.IsNotFound(err) {
			return r.Create(ctx, cm)
		}
		return err
	}

	existing.Data = cm.Data
	return r.Update(ctx, existing)
}

// generateCompleteConfig merges user-provided config with auto-generated settings
func (r *TermitePoolReconciler) generateCompleteConfig(pool *antflyaiv1alpha1.TermitePool) (string, error) {
	// Start with user config or empty object
	config := make(map[string]any)

	if pool.Spec.Config != "" {
		if err := json.Unmarshal([]byte(pool.Spec.Config), &config); err != nil {
			return "", fmt.Errorf("failed to parse spec.config: %w", err)
		}
	}

	// Build preload model list
	preload := make([]string, 0, len(pool.Spec.Models.Preload))
	for _, m := range pool.Spec.Models.Preload {
		name := m.Name
		if m.Variant != "" {
			name = name + ":" + m.Variant
		}
		preload = append(preload, name)
	}

	// Set auto-generated config (don't override if user specified)
	if _, exists := config["preload"]; !exists && len(preload) > 0 {
		config["preload"] = preload
	}

	// Build per-model loading strategies map
	// Only include models that have an explicit strategy override
	// Key format: "name" or "name-variant" (matches lazy registry naming)
	if _, exists := config["model_strategies"]; !exists {
		modelStrategies := make(map[string]string)
		for _, m := range pool.Spec.Models.Preload {
			if m.Strategy != "" {
				key := m.Name
				if m.Variant != "" {
					key = m.Name + "-" + m.Variant
				}
				modelStrategies[key] = string(m.Strategy)
			}
		}
		if len(modelStrategies) > 0 {
			config["model_strategies"] = modelStrategies
		}
	}

	// Set model directories based on models-dir default
	if _, exists := config["embedder_models_dir"]; !exists {
		config["embedder_models_dir"] = "/models/embedders"
	}
	if _, exists := config["chunker_models_dir"]; !exists {
		config["chunker_models_dir"] = "/models/chunkers"
	}
	if _, exists := config["reranker_models_dir"]; !exists {
		config["reranker_models_dir"] = "/models/rerankers"
	}

	// Set loading strategy config
	if pool.Spec.Models.LoadingStrategy != "" {
		switch pool.Spec.Models.LoadingStrategy {
		case antflyaiv1alpha1.LoadingStrategyLazy:
			// Lazy loading: set keep_alive if not specified
			if _, exists := config["keep_alive"]; !exists {
				if pool.Spec.Models.KeepAlive != nil {
					config["keep_alive"] = pool.Spec.Models.KeepAlive.Duration.String()
				} else {
					config["keep_alive"] = "5m" // Default 5 minutes
				}
			}
		case antflyaiv1alpha1.LoadingStrategyBounded:
			// Bounded loading: set max_loaded_models
			if _, exists := config["max_loaded_models"]; !exists {
				if pool.Spec.Models.MaxLoadedModels != nil {
					config["max_loaded_models"] = *pool.Spec.Models.MaxLoadedModels
				}
			}
			// Also set keep_alive for LRU eviction
			if _, exists := config["keep_alive"]; !exists {
				if pool.Spec.Models.KeepAlive != nil {
					config["keep_alive"] = pool.Spec.Models.KeepAlive.Duration.String()
				} else {
					config["keep_alive"] = "5m"
				}
			}
		}
		// Eager loading (default): models loaded at startup and never unloaded
	}

	// Marshal to JSON
	configJSON, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal config: %w", err)
	}

	return string(configJSON), nil
}

func (r *TermitePoolReconciler) reconcileStatefulSet(ctx context.Context, pool *antflyaiv1alpha1.TermitePool) error {
	replicas := pool.Spec.Replicas.Min

	// Build model list for init container pull command
	// Group models by variant to use --variants flag (backward compatible with older images)
	// Example: /termite pull --models-dir /models --variants i8 bge-small-en-v1.5 mxbai-rerank-base-v1
	variantGroups := make(map[string][]string) // variant -> []model names
	for _, m := range pool.Spec.Models.Preload {
		variant := m.Variant
		if variant == "" {
			variant = "f32" // default variant
		}
		variantGroups[variant] = append(variantGroups[variant], m.Name)
	}

	// Build pull command(s) - one per variant group, sorted for deterministic ordering
	variants := make([]string, 0, len(variantGroups))
	for v := range variantGroups {
		variants = append(variants, v)
	}
	slices.Sort(variants)

	var pullCmds []string
	for _, variant := range variants {
		names := variantGroups[variant]
		slices.Sort(names) // Sort model names too for consistency
		pullCmds = append(pullCmds, fmt.Sprintf("/termite pull --models-dir /models --variants %s %s",
			variant, strings.Join(names, " ")))
	}
	pullCmd := strings.Join(pullCmds, " && ")

	// Determine image
	image := r.TermiteImage
	if pool.Spec.Image != "" {
		image = pool.Spec.Image
	}

	sts := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pool.Name,
			Namespace: pool.Namespace,
			Labels:    r.labels(pool),
		},
		Spec: appsv1.StatefulSetSpec{
			ServiceName: pool.Name,
			Replicas:    &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: r.selectorLabels(pool),
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: r.labels(pool),
				},
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{
							Name:    "model-puller",
							Image:   image,
							Command: []string{"/bin/sh", "-c"},
							Args:    []string{pullCmd},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "models", MountPath: "/models"},
							},
							EnvFrom: []corev1.EnvFromSource{
								{ConfigMapRef: &corev1.ConfigMapEnvSource{
									LocalObjectReference: corev1.LocalObjectReference{Name: pool.Name + "-config"},
								}},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name:    "termite",
							Image:   image,
							Command: []string{"/termite"},
							Args:    []string{"run", "--config", "/config/config.json"},
							Ports: []corev1.ContainerPort{
								{Name: "http", ContainerPort: TermiteAPIPort, Protocol: corev1.ProtocolTCP},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "models", MountPath: "/models"},
								{Name: "config", MountPath: "/config", ReadOnly: true},
							},
							EnvFrom: []corev1.EnvFromSource{
								{ConfigMapRef: &corev1.ConfigMapEnvSource{
									LocalObjectReference: corev1.LocalObjectReference{Name: pool.Name + "-config"},
								}},
							},
							Resources: r.buildResources(pool),
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "models",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
						{
							Name: "config",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: pool.Name + "-config",
									},
									Items: []corev1.KeyToPath{
										{Key: "config.json", Path: "config.json"},
									},
								},
							},
						},
					},
					ImagePullSecrets: pool.Spec.ImagePullSecrets,
				},
			},
		},
	}

	// Apply GKE-specific pod configuration (Autopilot compute classes, spot instances, etc.)
	r.applyGKEPodSpec(&sts.Spec.Template, pool)

	// Add TPU node selector and tolerations (works in both Standard and Autopilot modes)
	// In Autopilot, TPU provisioning is triggered by these selectors, not by compute class
	if pool.Spec.Hardware.Accelerator != "" {
		if sts.Spec.Template.Spec.NodeSelector == nil {
			sts.Spec.Template.Spec.NodeSelector = make(map[string]string)
		}
		sts.Spec.Template.Spec.NodeSelector["cloud.google.com/gke-tpu-accelerator"] = pool.Spec.Hardware.Accelerator
		sts.Spec.Template.Spec.NodeSelector["cloud.google.com/gke-tpu-topology"] = pool.Spec.Hardware.Topology

		sts.Spec.Template.Spec.Tolerations = append(sts.Spec.Template.Spec.Tolerations, corev1.Toleration{
			Key:      "google.com/tpu",
			Operator: corev1.TolerationOpExists,
			Effect:   corev1.TaintEffectNoSchedule,
		})
	}

	// Add probes
	r.addProbes(sts, pool)

	// Set owner reference
	if err := ctrl.SetControllerReference(pool, sts, r.Scheme); err != nil {
		return err
	}

	// Add template hash annotation to trigger rolling updates when pod spec changes
	// This ensures pods are recreated when tolerations, resources, etc. change
	templateHash := computePodTemplateHash(&sts.Spec.Template)
	if sts.Spec.Template.Annotations == nil {
		sts.Spec.Template.Annotations = make(map[string]string)
	}
	sts.Spec.Template.Annotations["termite.antfly.io/template-hash"] = templateHash

	// Create or update
	existing := &appsv1.StatefulSet{}
	if err := r.Get(ctx, types.NamespacedName{Name: sts.Name, Namespace: sts.Namespace}, existing); err != nil {
		if errors.IsNotFound(err) {
			return r.Create(ctx, sts)
		}
		return err
	}

	// Update relevant fields
	existing.Spec.Replicas = sts.Spec.Replicas
	existing.Spec.Template = sts.Spec.Template
	return r.Update(ctx, existing)
}

// computePodTemplateHash computes a hash of the pod template spec.
// This is used to trigger rolling updates when the template changes.
func computePodTemplateHash(template *corev1.PodTemplateSpec) string {
	// Create a copy without the hash annotation itself to avoid circular dependency
	templateCopy := template.DeepCopy()
	delete(templateCopy.Annotations, "termite.antfly.io/template-hash")

	// Marshal to JSON for consistent hashing
	data, err := json.Marshal(templateCopy.Spec)
	if err != nil {
		// Fallback to empty hash on error (shouldn't happen)
		return ""
	}

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:8]) // Use first 8 bytes (16 hex chars)
}

func (r *TermitePoolReconciler) reconcilePDB(ctx context.Context, pool *antflyaiv1alpha1.TermitePool) error {
	// Get PDB configuration from either GKE config or Availability config
	var pdbConfig *antflyaiv1alpha1.PDBConfig

	// Prefer GKE PDB config if available
	if pool.Spec.GKE != nil && pool.Spec.GKE.PodDisruptionBudget != nil {
		pdbConfig = pool.Spec.GKE.PodDisruptionBudget
	} else if pool.Spec.Availability != nil && pool.Spec.Availability.PodDisruptionBudget != nil {
		pdbConfig = pool.Spec.Availability.PodDisruptionBudget
	}

	// If no PDB config or not enabled, skip
	if pdbConfig == nil || !pdbConfig.Enabled {
		return nil
	}

	pdbName := pool.Name + "-pdb"

	pdb := &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pdbName,
			Namespace: pool.Namespace,
		},
	}

	// Use CreateOrUpdate to ensure PDB is updated with latest configuration
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, pdb, func() error {
		// Set controller reference
		if err := ctrl.SetControllerReference(pool, pdb, r.Scheme); err != nil {
			return err
		}

		// Update PDB spec with labels
		pdb.Labels = r.labels(pool)

		// Set selector to match pool pods
		pdb.Spec.Selector = &metav1.LabelSelector{
			MatchLabels: r.selectorLabels(pool),
		}

		// Set MaxUnavailable or MinAvailable (prefer MaxUnavailable as recommended)
		if pdbConfig.MaxUnavailable != nil {
			maxUnavailable := intstr.FromInt(int(*pdbConfig.MaxUnavailable))
			pdb.Spec.MaxUnavailable = &maxUnavailable
			pdb.Spec.MinAvailable = nil // Clear MinAvailable when MaxUnavailable is set
		} else if pdbConfig.MinAvailable != nil {
			minAvailable := intstr.FromInt(int(*pdbConfig.MinAvailable))
			pdb.Spec.MinAvailable = &minAvailable
			pdb.Spec.MaxUnavailable = nil // Clear MaxUnavailable when MinAvailable is set
		} else {
			// Default to MaxUnavailable=1
			maxUnavailable := intstr.FromInt(1)
			pdb.Spec.MaxUnavailable = &maxUnavailable
			pdb.Spec.MinAvailable = nil
		}

		return nil
	})

	return err
}

// applyGKEPodSpec applies GKE-specific configuration to pod template spec
func (r *TermitePoolReconciler) applyGKEPodSpec(podTemplate *corev1.PodTemplateSpec, pool *antflyaiv1alpha1.TermitePool) {
	// GKE Autopilot mode: use compute class annotations
	if pool.Spec.GKE != nil && pool.Spec.GKE.Autopilot {
		// Initialize annotations if nil
		if podTemplate.Annotations == nil {
			podTemplate.Annotations = make(map[string]string)
		}

		// Set termination grace period for graceful shutdown
		gracePeriod := int64(15)
		podTemplate.Spec.TerminationGracePeriodSeconds = &gracePeriod

		// Check if this is a TPU workload - TPU workloads should NOT have a compute class
		// annotation because the TPU node selectors (gke-tpu-accelerator, gke-tpu-topology)
		// drive node provisioning directly. Adding a compute class like "Balanced" prevents
		// the cluster autoscaler from creating TPU nodes.
		isTPUWorkload := strings.Contains(pool.Spec.Hardware.Accelerator, "tpu")

		if isTPUWorkload {
			// For TPU workloads: don't set compute-class, let node selectors drive provisioning
			// The TPU node selectors are set in ensureTPUResources()

			// But DO add spot toleration if spot is requested
			// TPU spot nodes still have the cloud.google.com/gke-spot taint
			if pool.Spec.Hardware.Spot || pool.Spec.GKE.AutopilotComputeClass == "autopilot-spot" {
				podTemplate.Spec.Tolerations = append(podTemplate.Spec.Tolerations, corev1.Toleration{
					Key:      "cloud.google.com/gke-spot",
					Operator: corev1.TolerationOpEqual,
					Value:    "true",
					Effect:   corev1.TaintEffectNoSchedule,
				})
			}
			return
		}

		// For non-TPU workloads: apply compute class annotation
		computeClass := pool.Spec.GKE.AutopilotComputeClass
		if computeClass == "" {
			computeClass = "Balanced"
		}

		// Apply compute class annotation (required for GKE Autopilot non-TPU workloads)
		podTemplate.Annotations["cloud.google.com/compute-class"] = computeClass

		// Add spot toleration if using autopilot-spot compute class
		// GKE Autopilot spot nodes have the taint cloud.google.com/gke-spot=true:NoSchedule
		if computeClass == "autopilot-spot" {
			podTemplate.Spec.Tolerations = append(podTemplate.Spec.Tolerations, corev1.Toleration{
				Key:      "cloud.google.com/gke-spot",
				Operator: corev1.TolerationOpEqual,
				Value:    "true",
				Effect:   corev1.TaintEffectNoSchedule,
			})
		}

		return
	}

	// Standard GKE mode (non-Autopilot): use node selectors for spot instances
	if pool.Spec.Hardware.Spot {
		// Initialize nodeSelector if nil
		if podTemplate.Spec.NodeSelector == nil {
			podTemplate.Spec.NodeSelector = make(map[string]string)
		}

		// Apply Spot Nodes configuration using node selector
		podTemplate.Spec.NodeSelector["cloud.google.com/gke-spot"] = "true"

		// Set termination grace period for graceful shutdown on eviction
		gracePeriod := int64(15)
		podTemplate.Spec.TerminationGracePeriodSeconds = &gracePeriod
	}
}

func (r *TermitePoolReconciler) updateStatus(ctx context.Context, pool *antflyaiv1alpha1.TermitePool) error {
	// Get StatefulSet to read replica status
	sts := &appsv1.StatefulSet{}
	if err := r.Get(ctx, types.NamespacedName{Name: pool.Name, Namespace: pool.Namespace}, sts); err != nil {
		if !errors.IsNotFound(err) {
			return err
		}
		pool.Status.Phase = antflyaiv1alpha1.TermitePoolPhasePending
	} else {
		pool.Status.Replicas.Ready = sts.Status.ReadyReplicas
		pool.Status.Replicas.Total = sts.Status.Replicas
		pool.Status.Replicas.Desired = *sts.Spec.Replicas

		if sts.Status.ReadyReplicas == *sts.Spec.Replicas {
			pool.Status.Phase = antflyaiv1alpha1.TermitePoolPhaseRunning
		} else if sts.Status.ReadyReplicas > 0 {
			pool.Status.Phase = antflyaiv1alpha1.TermitePoolPhaseScaling
		} else {
			pool.Status.Phase = antflyaiv1alpha1.TermitePoolPhasePending
		}
	}

	return r.Status().Update(ctx, pool)
}

func (r *TermitePoolReconciler) labels(pool *antflyaiv1alpha1.TermitePool) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":      "termite",
		"app.kubernetes.io/component": "termite-pool",
		"app.kubernetes.io/instance":  pool.Name,
		"antfly.io/pool":              pool.Name,
		"antfly.io/workload-type":     string(pool.Spec.WorkloadType),
	}
}

func (r *TermitePoolReconciler) selectorLabels(pool *antflyaiv1alpha1.TermitePool) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":     "termite",
		"app.kubernetes.io/instance": pool.Name,
		"antfly.io/pool":             pool.Name,
	}
}

func (r *TermitePoolReconciler) buildResources(pool *antflyaiv1alpha1.TermitePool) corev1.ResourceRequirements {
	// If user provided explicit resources, use those
	if pool.Spec.Resources != nil {
		resources := pool.Spec.Resources.DeepCopy()
		// Ensure TPU resources are set if accelerator is configured
		r.ensureTPUResources(resources, pool)
		return *resources
	}

	// Build default resources
	resources := corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceMemory: resource.MustParse("4Gi"),
			corev1.ResourceCPU:    resource.MustParse("1"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceMemory: resource.MustParse("8Gi"),
			corev1.ResourceCPU:    resource.MustParse("2"),
		},
	}

	// Add TPU resources if accelerator is configured
	r.ensureTPUResources(&resources, pool)

	return resources
}

// ensureTPUResources adds google.com/tpu resource requests/limits if an accelerator is configured
// and TPU resources are not already specified. This is required for GKE Autopilot.
func (r *TermitePoolReconciler) ensureTPUResources(resources *corev1.ResourceRequirements, pool *antflyaiv1alpha1.TermitePool) {
	// Only add TPU resources if accelerator is configured
	if pool.Spec.Hardware.Accelerator == "" {
		return
	}

	tpuResourceName := corev1.ResourceName("google.com/tpu")

	// Calculate TPU count from topology (e.g., "2x2" = 4, "2x4" = 8, "4x4" = 16)
	tpuCount := calculateTPUCountFromTopology(pool.Spec.Hardware.Topology)

	// Initialize maps if nil
	if resources.Requests == nil {
		resources.Requests = corev1.ResourceList{}
	}
	if resources.Limits == nil {
		resources.Limits = corev1.ResourceList{}
	}

	// Add TPU to requests if not already present
	if _, exists := resources.Requests[tpuResourceName]; !exists {
		resources.Requests[tpuResourceName] = *resource.NewQuantity(int64(tpuCount), resource.DecimalSI)
	}

	// Add TPU to limits if not already present
	if _, exists := resources.Limits[tpuResourceName]; !exists {
		resources.Limits[tpuResourceName] = *resource.NewQuantity(int64(tpuCount), resource.DecimalSI)
	}
}

// calculateTPUCountFromTopology parses a topology string like "2x2" and returns the TPU count
func calculateTPUCountFromTopology(topology string) int {
	if topology == "" {
		return 4 // Default to 2x2 = 4
	}

	// Parse "NxM" format
	parts := strings.Split(topology, "x")
	if len(parts) != 2 {
		return 4 // Default fallback
	}

	var rows, cols int
	if _, err := fmt.Sscanf(parts[0], "%d", &rows); err != nil {
		return 4
	}
	if _, err := fmt.Sscanf(parts[1], "%d", &cols); err != nil {
		return 4
	}

	return rows * cols
}

func (r *TermitePoolReconciler) addProbes(sts *appsv1.StatefulSet, pool *antflyaiv1alpha1.TermitePool) {
	container := &sts.Spec.Template.Spec.Containers[0]

	// Default startup probe (allows 5 min for model loading)
	failureThreshold := int32(30)
	periodSeconds := int32(10)

	if pool.Spec.Availability != nil && pool.Spec.Availability.StartupProbe != nil {
		if pool.Spec.Availability.StartupProbe.FailureThreshold != nil {
			failureThreshold = *pool.Spec.Availability.StartupProbe.FailureThreshold
		}
		if pool.Spec.Availability.StartupProbe.PeriodSeconds != nil {
			periodSeconds = *pool.Spec.Availability.StartupProbe.PeriodSeconds
		}
	}

	container.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/api/models",
				Port: intstr.FromString("http"),
			},
		},
		FailureThreshold: failureThreshold,
		PeriodSeconds:    periodSeconds,
	}

	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/api/models",
				Port: intstr.FromString("http"),
			},
		},
		PeriodSeconds: 5,
	}

	container.LivenessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/api/models",
				Port: intstr.FromString("http"),
			},
		},
		PeriodSeconds: 30,
	}
}

// validatePool performs controller-level validation (fallback when webhook is disabled)
func (r *TermitePoolReconciler) validatePool(pool *antflyaiv1alpha1.TermitePool) error {
	// Validate GKE config
	if pool.Spec.GKE != nil {
		gke := pool.Spec.GKE

		// Validate compute class requires Autopilot
		if gke.AutopilotComputeClass != "" && !gke.Autopilot {
			return fmt.Errorf("spec.gke.autopilotComputeClass is set but spec.gke.autopilot=false; compute classes only work with GKE Autopilot clusters")
		}

		// Validate compute class enum (only if non-empty)
		if gke.AutopilotComputeClass != "" {
			validClasses := map[string]bool{
				"Accelerator": true, "Balanced": true, "Performance": true,
				"Scale-Out": true, "autopilot": true, "autopilot-spot": true,
			}
			if !validClasses[gke.AutopilotComputeClass] {
				return fmt.Errorf("invalid GKE Autopilot compute class '%s'", gke.AutopilotComputeClass)
			}
		}

		// Validate no conflicting settings (spot + autopilot)
		// Exception: TPU workloads CAN use hardware.spot=true even in Autopilot mode
		// because TPU provisioning doesn't use compute class (node selectors drive it)
		isTPUWorkload := strings.Contains(pool.Spec.Hardware.Accelerator, "tpu")
		if gke.Autopilot && pool.Spec.Hardware.Spot && !isTPUWorkload {
			return fmt.Errorf("spec.hardware.spot=true conflicts with spec.gke.autopilot=true; use gke.autopilotComputeClass='autopilot-spot' instead")
		}
	}

	// Validate replica counts
	if pool.Spec.Replicas.Min < 0 {
		return fmt.Errorf("spec.replicas.min must be >= 0, got %d", pool.Spec.Replicas.Min)
	}
	if pool.Spec.Replicas.Max <= 0 {
		return fmt.Errorf("spec.replicas.max must be > 0, got %d", pool.Spec.Replicas.Max)
	}
	if pool.Spec.Replicas.Min > pool.Spec.Replicas.Max {
		return fmt.Errorf("spec.replicas.min (%d) cannot be greater than spec.replicas.max (%d)",
			pool.Spec.Replicas.Min, pool.Spec.Replicas.Max)
	}

	return nil
}

// SetupWithManager sets up the controller with the Manager
func (r *TermitePoolReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&antflyaiv1alpha1.TermitePool{}).
		Owns(&appsv1.StatefulSet{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&policyv1.PodDisruptionBudget{}).
		Complete(r)
}
