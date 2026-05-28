

/*
 *  aeif_cond_alpha_alt_neuron_nestml.cu
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <config.h>
#include <cmath>
#include <iostream>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>
#include <boost/ref.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "aeif_cond_alpha_alt_neuron_nestml.h"
#include "spike_buffer.h"

using namespace aeif_cond_alpha_alt_neuron_nestml_ns;


/*
 * Lightweight context object passed to generated NESTML code.
 *
 * It stores runtime information that may be needed by generated
 * derivative or event-handling code.
 */
struct aeif_cond_alpha_alt_neuron_nestml_odeint_context
{
  int i_node_0_;
};


namespace
{

typedef float value_type;
typedef thrust::device_vector<value_type> ode_state_type;


/*
 * Copy scalar NEST GPU state variables into a compact Odeint state vector.
 *
 * NEST layout: var_arr[i_neuron * n_var + scalar_index]
 * Odeint layout: ode_state[i_neuron * N_SCAL_VAR + scalar_index]
 *
 * Port variables are intentionally not copied.
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_CopyNestToOdeState(
    int n_node,
    const float* var_arr,
    int n_var,
    float* ode_state)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_neuron >= n_node)
    return;

  const float* src = var_arr + n_var * i_neuron;
  float* dst = ode_state + N_SCAL_VAR * i_neuron;

  for (int i = 0; i < N_SCAL_VAR; ++i)
    dst[i] = src[i];
}


/*
 * Copy the compact Odeint scalar state back into the NEST GPU state array.
 *
 * Port variables remain untouched here.
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_CopyOdeStateToNest(
    int n_node,
    float* var_arr,
    int n_var,
    const float* ode_state)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_neuron >= n_node)
    return;

  float* dst = var_arr + n_var * i_neuron;
  const float* src = ode_state + N_SCAL_VAR * i_neuron;

  for (int i = 0; i < N_SCAL_VAR; ++i)
    dst[i] = src[i];


  if (dst[i_refr_t] < 0.0f)
    dst[i_refr_t] = 0.0f;

}

} // anonymous namespace


namespace aeif_cond_alpha_alt_neuron_nestml_odeint_system_ns
{

using namespace aeif_cond_alpha_alt_neuron_nestml_ns;


/*
 * Generated NESTML derivative functions.
 *
 * These functions are generated from all unique integrate_odes(...) calls.
 *
 * Impo:
 * NumericDifferentiationFunction.jinja2 must be rendered with gsl_printer,
 * otherwise expressions may be generated with var[...] instead of y[...].
 */

template<int NVAR, int NPARAM>
__device__
void Derivatives_g_exc_g_inh_refr_t_w(
    double x,
    float *y,
    float *dydx,
    float *param,
    aeif_cond_alpha_alt_neuron_nestml_odeint_context data_struct)
{  

    dydx[i_g_exc] =1.0 * y[i_g_exc__d];
      
    dydx[i_g_exc__d] =(-y[i_g_exc]) / pow(param[i_tau_syn_exc], 2) - 2 * y[i_g_exc__d] / param[i_tau_syn_exc];
      
    dydx[i_g_inh] =1.0 * y[i_g_inh__d];
      
    dydx[i_g_inh__d] =(-y[i_g_inh]) / pow(param[i_tau_syn_inh], 2) - 2 * y[i_g_inh__d] / param[i_tau_syn_inh];
      
    dydx[i_V_m] =0;
      
    dydx[i_w] =param[i_a] * ((-param[i_E_L]) / param[i_tau_w] + min(y[i_V_m], param[i_V_peak]) / param[i_tau_w]) - y[i_w] / param[i_tau_w];
      
    dydx[i_refr_t] =(-1.0);
      
}
template<int NVAR, int NPARAM>
__device__
void Derivatives_V_m_g_exc_g_inh_w(
    double x,
    float *y,
    float *dydx,
    float *param,
    aeif_cond_alpha_alt_neuron_nestml_odeint_context data_struct)
{  

    dydx[i_g_exc] =1.0 * y[i_g_exc__d];
      
    dydx[i_g_exc__d] =(-y[i_g_exc]) / pow(param[i_tau_syn_exc], 2) - 2 * y[i_g_exc__d] / param[i_tau_syn_exc];
      
    dydx[i_g_inh] =1.0 * y[i_g_inh__d];
      
    dydx[i_g_inh__d] =(-y[i_g_inh]) / pow(param[i_tau_syn_inh], 2) - 2 * y[i_g_inh__d] / param[i_tau_syn_inh];
      
    dydx[i_V_m] =param[i_g_L] * (param[i_Delta_T] * exp(((-param[i_V_th]) + min(y[i_V_m], param[i_V_peak])) / param[i_Delta_T]) / param[i_C_m] + param[i_E_L] / param[i_C_m] - min(y[i_V_m], param[i_V_peak]) / param[i_C_m]) + param[i_E_exc] * y[i_g_exc] / param[i_C_m] + param[i_E_inh] * y[i_g_inh] / param[i_C_m] + param[i_I_e] / param[i_C_m] + param[i_I_stim] / param[i_C_m] + ((-y[i_g_exc]) * min(y[i_V_m], param[i_V_peak]) - y[i_g_inh] * min(y[i_V_m], param[i_V_peak]) - y[i_w]) / param[i_C_m];
      
    dydx[i_w] =param[i_a] * ((-param[i_E_L]) / param[i_tau_w] + min(y[i_V_m], param[i_V_peak]) / param[i_tau_w]) - y[i_w] / param[i_tau_w];
      
    dydx[i_refr_t] =0;
      
}



/*
 * Dispatch to the correct derivative function depending on the original
 * NESTML control flow around integrate_odes().
 */
template<int NVAR, int NPARAM>
__device__
void OdeintDerivatives(
    double x,
    float* y,
    float* dydx,
    float* param,
    aeif_cond_alpha_alt_neuron_nestml_odeint_context data_struct)
{

  if (y[i_refr_t] > 0)
  {
    aeif_cond_alpha_alt_neuron_nestml_odeint_system_ns::Derivatives_g_exc_g_inh_refr_t_w<NVAR, NPARAM>(
        x,
        y,
        dydx,
        param,
        data_struct);
  }

  else
  {
    aeif_cond_alpha_alt_neuron_nestml_odeint_system_ns::Derivatives_V_m_g_exc_g_inh_w<NVAR, NPARAM>(
        x,
        y,
        dydx,
        param,
        data_struct);
  }

}

} // namespace aeif_cond_alpha_alt_neuron_nestml_odeint_system_ns


namespace
{

/*
 * Device functor for the right-hand side of the ODE system.
 *
 * Odeint calls aeif_cond_alpha_alt_neuron_nestml_OdeintSystem on the host.
 * That system function launches this Thrust functor on the GPU.
 */
struct aeif_cond_alpha_alt_neuron_nestml_DerivativeFunctor
{
  const float* x;
  float* dxdt;
  const float* param;
  int param_stride;

  __device__
  void operator()(int i_neuron) const
  {
    constexpr int NVAR   = N_SCAL_VAR;
    constexpr int NPARAM = N_SCAL_PARAM;

    const float* y_src = x + i_neuron * N_SCAL_VAR;
    float* dydt = dxdt + i_neuron * N_SCAL_VAR;
    const float* p = param + i_neuron * param_stride;

    /*
     * The generated derivative code expects a mutable y array.
     * Therefore we copy the compact scalar state into a local array.
     */
    float y[NVAR];

    for (int i = 0; i < NVAR; ++i)
      y[i] = y_src[i];

    aeif_cond_alpha_alt_neuron_nestml_odeint_context data_struct;
    data_struct.i_node_0_ = 0;

    aeif_cond_alpha_alt_neuron_nestml_odeint_system_ns::OdeintDerivatives<NVAR, NPARAM>(
        0.0,
        y,
        dydt,
        const_cast<float*>(p),
        data_struct);
  }
};


/*
 * Boost.Odeint system function.
 *
 * x: compact scalar state vector on GPU
 * dxdt:compact derivative vector on GPU
 */
struct aeif_cond_alpha_alt_neuron_nestml_OdeintSystem
{
  const float* param;
  int param_stride;
  int n_neuron;

  void operator()(const ode_state_type& x,
                  ode_state_type& dxdt,
                  const value_type /*t*/) const
  {
    const float* x_ptr = thrust::raw_pointer_cast(x.data());
    float* dxdt_ptr = thrust::raw_pointer_cast(dxdt.data());

    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(n_neuron);

    aeif_cond_alpha_alt_neuron_nestml_DerivativeFunctor functor;
    functor.x = x_ptr;
    functor.dxdt = dxdt_ptr;
    functor.param = param;
    functor.param_stride = param_stride;

    thrust::for_each(thrust::device, begin, end, functor);
  }
};

} // anonymous namespace


/*
 * PRE-INTEGRATION KERNEL
 *
 * Handles onReceive before ODE integration.
 * Incoming spikes affect the ODE state before the Boost.Odeint step.
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_PreUpdate(
    int n_node,
    int i_node_0,
    float* var_arr,
    float* param_arr,
    int n_var,
    int n_param)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_neuron >= n_node)
    return;

  float* y     = var_arr   + n_var   * i_neuron;
  float* var   = y;
  float* param = param_arr + n_param * i_neuron;

  double x = 0.0;
  bool end_time_step = true;

  aeif_cond_alpha_alt_neuron_nestml_odeint_context data_struct;
  data_struct.i_node_0_ = i_node_0;

  /*
   * Begin NESTML generated code for onReceive block(s).
   */

  if (y[N_SCAL_VAR + i_exc_spikes] != 0.0f)
  {    
    y[i_g_exc__d] += (0.001 * var[N_SCAL_VAR + i_exc_spikes]) * (M_E / param[i_tau_syn_exc]) * 1.0 * 1000.0;
    y[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  }


  if (y[N_SCAL_VAR + i_inh_spikes] != 0.0f)
  {    
    y[i_g_inh__d] += (0.001 * var[N_SCAL_VAR + i_inh_spikes]) * (M_E / param[i_tau_syn_inh]) * 1.0 * 1000.0;
    y[N_SCAL_VAR + i_inh_spikes] = 0.0f;
  }


}


/*
 * POST-INTEGRATION KERNEL
 *
 * Handles remaining update-block statements, threshold, reset,
 * spike emission and safety reset of spike input ports.
 * onReceive is handled before integration in aeif_cond_alpha_alt_neuron_nestml_PreUpdate().
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_PostUpdate(
    int n_node,
    int i_node_0,
    float* var_arr,
    float* param_arr,
    int n_var,
    int n_param)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_neuron >= n_node)
    return;

  float* y     = var_arr   + n_var   * i_neuron;
  float* var   = y;
  float* param = param_arr + n_param * i_neuron;

  double x = 0.0;
  bool end_time_step = true;

  aeif_cond_alpha_alt_neuron_nestml_odeint_context data_struct;
  data_struct.i_node_0_ = i_node_0;

  /*
   * Begin NESTML generated code for update block(s).
   *
   * integrate_odes() itself has already been handled by Boost.Odeint.
   * The generated Block.jinja2 code normally leaves only comments for integrate_odes() and keeps additional statements.
   */
  if (y[i_refr_t] > 0)
  {  

    // start rendered code for integrate_odes(g_exc, g_inh, refr_t, w)
  }
  else
  {  

    // start rendered code for integrate_odes(V_m, g_exc, g_inh, w)
  }

  /*
   * Begin NESTML generated code for onCondition block(s).
   */


  if (y[i_refr_t] <= 0 && y[i_V_m] >= param[i_V_peak])
  {    
    y[i_refr_t] = param[i_refr_T];
    y[i_V_m] = param[i_V_reset];
    y[i_w] += param[i_b];
    PushSpike(i_node_0 + i_neuron, 1.0);;
  }




  /*
   * Safety reset of all spike input port variables.
   * Normally they are already reset in PreUpdate.
   */

  y[N_SCAL_VAR + i_exc_spikes] = 0.0f;

  y[N_SCAL_VAR + i_inh_spikes] = 0.0f;

}



// Class methods

aeif_cond_alpha_alt_neuron_nestml::~aeif_cond_alpha_alt_neuron_nestml()
{
  Free();
}


int aeif_cond_alpha_alt_neuron_nestml::Init(int i_node_0,
                           int n_node,
                           int /*n_port*/,
                           int i_group,
                           unsigned long long* seed)
{
  BaseNeuron::Init(
      i_node_0,
      n_node,
      2 /*n_port*/,
      i_group,
      seed);

  node_type_ = i_aeif_cond_alpha_alt_neuron_nestml_model;


  /*
   * State variables
   */
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_var_      = n_scal_var_ + n_port_var_;


  /*
   * Parameters
   */
  n_scal_param_ = N_SCAL_PARAM;
  n_param_      = n_scal_param_;


  AllocParamArr();
  AllocVarArr();


  scal_var_name_   = aeif_cond_alpha_alt_neuron_nestml_scal_var_name;
  scal_param_name_ = aeif_cond_alpha_alt_neuron_nestml_scal_param_name;
  port_var_name_   = aeif_cond_alpha_alt_neuron_nestml_port_var_name;


  /*
   * Parameters
   */

  SetScalParam(
      0,
      n_node,
      "C_m",
      
  281.0);  // as pF

  SetScalParam(
      0,
      n_node,
      "refr_T",
      
  2);  // as ms

  SetScalParam(
      0,
      n_node,
      "V_reset",
      
  (-60.0));  // as mV

  SetScalParam(
      0,
      n_node,
      "g_L",
      
  30.0);  // as nS

  SetScalParam(
      0,
      n_node,
      "E_L",
      
  (-70.6));  // as mV

  SetScalParam(
      0,
      n_node,
      "a",
      
  4);  // as nS

  SetScalParam(
      0,
      n_node,
      "b",
      
  80.5);  // as pA

  SetScalParam(
      0,
      n_node,
      "Delta_T",
      
  2.0);  // as mV

  SetScalParam(
      0,
      n_node,
      "tau_w",
      
  144.0);  // as ms

  SetScalParam(
      0,
      n_node,
      "V_th",
      
  (-50.4));  // as mV

  SetScalParam(
      0,
      n_node,
      "V_peak",
      
  0);  // as mV

  SetScalParam(
      0,
      n_node,
      "tau_syn_exc",
      
  0.2);  // as ms

  SetScalParam(
      0,
      n_node,
      "tau_syn_inh",
      
  2.0);  // as ms

  SetScalParam(
      0,
      n_node,
      "E_exc",
      
  0);  // as mV

  SetScalParam(
      0,
      n_node,
      "E_inh",
      
  (-85.0));  // as mV

  SetScalParam(
      0,
      n_node,
      "I_e",
      
  0);  // as pA


  /*
   * Internal variables
   */

  SetScalParam(
      0,
      n_node,
      "__h",
      0.0);


  /*
   * Continuous input ports
   */

  SetScalParam(
      0,
      n_node,
      "I_stim",
      0.0);


  /*
   * State variables
   */

  SetScalVar(
      0,
      n_node,
      "V_m",
      
  *GetScalParam(0, n_node, "E_L"));  // as mV

  SetScalVar(
      0,
      n_node,
      "w",
      
  0);  // as pA

  SetScalVar(
      0,
      n_node,
      "refr_t",
      
  0);  // as ms

  SetScalVar(
      0,
      n_node,
      "g_exc",
      
  0);  // as nS

  SetScalVar(
      0,
      n_node,
      "g_exc__d",
      
  0);  // as nS / ms

  SetScalVar(
      0,
      n_node,
      "g_inh",
      
  0);  // as nS

  SetScalVar(
      0,
      n_node,
      "g_inh__d",
      
  0);  // as nS / ms



  /*
   * Compact state buffer for Boost.Odeint.
   * Size = n_node_ * N_SCAL_VAR
   */
  ode_state_ = new thrust::device_vector<float>(
      static_cast<size_t>(n_node_) * static_cast<size_t>(N_SCAL_VAR));



  /*
   * Multiplication factor of input signal is always 1 for all nodes.
   */
  float input_weight = 1.0f;

  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));

  gpuErrchk(cudaMemcpy(
      port_weight_arr_,
      &input_weight,
      sizeof(float),
      cudaMemcpyHostToDevice));

  port_weight_arr_step_  = 0;
  port_weight_port_step_ = 0;


  /*
   * Process the input spikes.
   */
  port_input_arr_ =
      GetVarArr() + n_scal_var_ + GetPortVarIdx("exc_spikes");

  port_input_arr_step_ =
      n_var_;

  port_input_port_step_ =
      1;



  return 0;
}


int aeif_cond_alpha_alt_neuron_nestml::Calibrate(double /*time_min*/,
                                float /*time_resolution*/)
{
  /*
   * Boost.Odeint path does not need legacy calibration.
   */
  return 0;
}


int aeif_cond_alpha_alt_neuron_nestml::Update(long long /*it*/, double t1)
{
  float dt = 0.0f;

  gpuErrchk(cudaMemcpyFromSymbol(
      &dt,
      NESTGPUTimeResolution,
      sizeof(float)));

  const float t0    = static_cast<float>(t1) - dt;
  const float t_end = static_cast<float>(t1);


  /*
   * Apply incoming spikes before ODE integration.
   */
  aeif_cond_alpha_alt_neuron_nestml_PreUpdate
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          i_node_0_,
          var_arr_,
          param_arr_,
          n_var_,
          n_param_);

  gpuErrchk(cudaPeekAtLastError());


  float* ode_state_ptr = thrust::raw_pointer_cast(ode_state_->data());


  /*
   * Copy current NEST GPU scalar state into compact Odeint state.
   *
   * This is done after PreUpdate, so incoming spikes already affected
   * the scalar ODE state before integration starts.
   */
  aeif_cond_alpha_alt_neuron_nestml_CopyNestToOdeState
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          var_arr_,
          n_var_,
          ode_state_ptr);

  gpuErrchk(cudaPeekAtLastError());


  /*
   * Built-in Boost.Odeint Dormand-Prince adaptive stepper.
   */
  typedef boost::numeric::odeint::runge_kutta_dopri5<
      ode_state_type,
      value_type,
      ode_state_type,
      value_type,
      boost::numeric::odeint::thrust_algebra,
      boost::numeric::odeint::thrust_operations> stepper_type;


  aeif_cond_alpha_alt_neuron_nestml_OdeintSystem system;
  system.param        = param_arr_;
  system.param_stride = n_param_;
  system.n_neuron     = n_node_;


  /*
   * Initial internal Odeint step size.
   *
   * Built-in aeif_cond_alpha uses h0_rel = 1.0e-2 by default.
   */
  const float odeint_h0 = dt * 1.0e-2f;


  boost::numeric::odeint::integrate_adaptive(
      boost::numeric::odeint::make_controlled(
          1.0e-5f,  // absolute tolerance
          1.0e-4f,  // relative tolerance
          stepper_type()),
      boost::ref(system),
      *ode_state_,
      t0,
      t_end,
      odeint_h0);


  /*
   * Copy integrated scalar state back to normal NEST GPU state array.
   * Port variables are not touched here.
   */
  aeif_cond_alpha_alt_neuron_nestml_CopyOdeStateToNest
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          var_arr_,
          n_var_,
          ode_state_ptr);

  gpuErrchk(cudaPeekAtLastError());


  /*
   * Handle threshold, reset and PushSpike after ODE integration.
   */
  aeif_cond_alpha_alt_neuron_nestml_PostUpdate
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          i_node_0_,
          var_arr_,
          param_arr_,
          n_var_,
          n_param_);

  gpuErrchk(cudaPeekAtLastError());


  return 0;
}


int aeif_cond_alpha_alt_neuron_nestml::Free()
{
  delete ode_state_;
  ode_state_ = nullptr;

  FreeVarArr();
  FreeParamArr();

  return 0;
}