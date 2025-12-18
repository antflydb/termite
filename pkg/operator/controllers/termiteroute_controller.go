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

import (
	"context"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	antflyaiv1alpha1 "github.com/antflydb/termite/pkg/operator/api/v1alpha1"
)

// TermiteRouteReconciler reconciles a TermiteRoute object
type TermiteRouteReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=antfly.io,resources=termiteroutes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=antfly.io,resources=termiteroutes/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=antfly.io,resources=termiteroutes/finalizers,verbs=update

// Reconcile handles TermiteRoute reconciliation
func (r *TermiteRouteReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Fetch the TermiteRoute
	route := &antflyaiv1alpha1.TermiteRoute{}
	if err := r.Get(ctx, req.NamespacedName, route); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("TermiteRoute not found, ignoring")
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	logger.Info("Reconciling TermiteRoute", "name", route.Name)

	// Validate referenced pools exist
	for _, dest := range route.Spec.Route {
		pool := &antflyaiv1alpha1.TermitePool{}
		if err := r.Get(ctx, client.ObjectKey{Name: dest.Pool, Namespace: route.Namespace}, pool); err != nil {
			if errors.IsNotFound(err) {
				logger.Error(err, "Referenced pool not found", "pool", dest.Pool)
				// Update status to indicate invalid configuration
				route.Status.Active = false
				if err := r.Status().Update(ctx, route); err != nil {
					return ctrl.Result{}, err
				}
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, err
		}
	}

	// Route is valid, mark as active
	route.Status.Active = true
	if err := r.Status().Update(ctx, route); err != nil {
		return ctrl.Result{}, err
	}

	// The actual route configuration is applied by the proxy
	// which watches TermiteRoute resources directly.
	// The operator's role is primarily validation and status management.

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *TermiteRouteReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&antflyaiv1alpha1.TermiteRoute{}).
		Complete(r)
}
