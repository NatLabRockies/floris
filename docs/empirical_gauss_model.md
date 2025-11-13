
(empirical_gauss_model)=
# Empirical Gaussian model

FLORIS's "empirical" model has the same Gaussian wake shape as other popular
FLORIS models. However, the models that describe the wake width and deflection
have been reorganized to provide simpler tuning and data fitting.

## Wake shape

The velocity deficit at a point $(x, y, z)$ in the wake follows a Gaussian
curve, i.e.,

$$ \frac{u}{U_\infty} = 1 - Ce^{-\frac{(y-\delta_y)^2}{2\sigma_y^2} -\frac{(z-z_h-\delta_z)^2}{2\sigma_z^2}} $$

where the $(x, y, z)$ origin is at the turbine location (at ground level).
The terms $C$, $\sigma_y$, $\sigma_z$, $\delta_y$, and $\delta_z$ all depend
on the downstream location $x$.

$C$ is the scaling factor for the Gaussian curve, defined as

$$C = \frac{1}{8\sigma_{0_D}^2}\left(1 - \sqrt{1 - \frac{\sigma_{y0} \sigma_{z0} C_T}{\sigma_y \sigma_z}}\right)$$

Here, $C_T$ is the turbine thrust coefficient, which includes any reduction
in thrust due to yaw or tilt of the turbine rotor. $\sigma_{y0}$ and
$\sigma_{z0}$ define the wake width at the turbine location $x=0$, which are
based on the user-specified rotor-diameter normalized initial width
$\sigma_{0_D}$. Note that
this contrasts with FLORIS's
other Gaussian models, where $\sigma_{y0}$ and $\sigma_{z0}$ are defined at
the end of the near wake/beginning of the far wake, at some $x_0 > 0$. The
normalization term $8\sigma_{0_D}^2$ provides consistency with actuator
disc theory.

## Wake expansion
The wake lateral and vertical widths, $\sigma_y$ and $\sigma_z$, respectively,
are a function of downstream distance $x$. The expansion of the wake is
described by a user-tunable, piecewise linear function. This is simplest to
express as an integral of a piecewise constant wake expansion rate $k$, i.e.,

$$ \sigma_{y}(x) = \int_{0}^x \sum_{i=0}^n k_i \mathbf{1}_{[b_{i}, b_{i+1})} (x') dx' + \sigma_{y0} $$

Here, $\mathbf{1}_{[a, b)}(x)$ is the indicator function, which takes value
1 when $a \leq x < b$, and 0 otherwise.
The above function ensures that expansion rate $k_i$ applies only between
breakpoints $b_{i-1}$ and $b_i$, allowing $n+1$ varying rates of linear
expansion at different downstream ranges, determined by the $b_i$. Note that
$b_0 = 0$ and $b_{n+1} = \infty$ by design.

A slight modification is made to the above so that the wake width varies
smoothly. As stated above, the wake expansion rate contains jump
discontinuities that create "sharp" changes in the wake width. To avoid this,
the indicator function $\mathbf{1}_{[a, b)}(x)$ is replaced with a pair of
"smoothstep" functions that vary smoothly with width parameter $d$. In the
limit as $d\rightarrow 0$, the approximation becomes exact.

While the form of this wake expansion model seems complex, it is very simple
to tune to fit data: the user provides the $n+1$ expansion rates
$k_i, i=0,\dots,n+1$
(defined as a list in the `wake_expansion_rates` field of the input yaml)
and
the $n$ 'break points' $b_i, i=1,\dots,n$ where those expansion rates should go
into effect (specified in terms of rotor diameters downstream as a list in the
`breakpoints_D` field of the input yaml.

As well as these, the initial width $\sigma_{0_D}$ should be
provided by setting `sigma_0_D` and the
logistic function width $d$ as `smoothing_length_D` (both specified in
terms of rotor diameters).

We expect that the default values for $\sigma_{0_D}$ and $d$ should be
satisfactory for most users. Further, we anticipate that most users will not
need more than $n+1=3$ expansion rates (along with $n=2$ break points) to
describe the wake expansion.

## Wake deflection

The deflection of the wake centerline $\delta_y$ and $\delta_z$ due to
yawing and tilting, respectively, follow a simple model

$$ \delta = k_\text{def} C_T \alpha \operatorname{ln}\left(\frac{x/D - c}{x/D + c} + 2\right)$$

Here, $k_\text{def}$ is a user-tunable deflection gain and $\alpha$ is the
misalignment. When computing the lateral wake deflection $\delta_y$ due to
yaw misalignment, $\alpha$ should be the yaw misalignment _specified in
radians, clockwise positive from the wind direction_. When
computing the vertical wake deflection $\delta_z$ due to rotor tilt,
$\alpha$ should be the tilt angle _specified in radians, clockwise positive
when the rotor is tilted back_.

Finally, $c$ in the above deflection model is a 'deflection rate'. This
specifies how quickly the wake will reach it's maximum deflection
$k_\text{def} C_T \alpha \operatorname{ln}(3)$ for a given
yaw/tilt angle.

User-tunable parameters of the model are as follows:
- A separately tunable deflection gain $k_\text{def}$ for each of
lateral deflections (due to yaw misalignments) and vertical deflections
(due to nonzero tilt), specified using `horizontal_deflection_gain_D` and
`vertical_deflection_gain_D` (specified in terms of rotor diameters)
- The deflection rate $c$, specified using `deflection_rate`.

We anticipate that most users will be able to use the default value for $c$,
and set `vertical_deflection_gain_D` to the same value as
`horizontal_deflection_gain_D` (which can also be achieved by providing
`vertical_deflection_gain_D = -1`).

## Wake-induced mixing

Finally, turbines contribute to mixing in the flow. In other models, this
extra mixing is accounted for by adding to the turbulence intensity value. In
the empirical model, explicit dependencies on turbulence intensity are removed
completely to aid in tuning. Instead, a non-physical "wake-induced mixing
factor" is specified for turbine $j$ as

$$ \text{WIM}_j = \sqrt{\sum_{i \neq j}
\left(\frac{A_{ij} a_i} {\bar{d}_{ij}^2}\right)^2} $$

where $A_{ij}$ is the area of overlap of the wake of turbine $i$
onto turbine $j$; $a_i$ is the axial induction factor of the
turbine $i$;
and $\bar{d}_{ij} = (x_j - x_i)/D_i$ is the downstream distance of turbine $j$ from
the turbine $i$, normalized by turbine $i$'s rotor diameter. Here, $A_{ij} = 1$ if turbine $j$ is fully covered by turbine $i$'s wake; $A_{ij} = 0$ if turbine $j$ is not covered at all by turbine $i$'s wake; and $0 < A_{ij} < 1$ if turbine $j$ is partially covered by turbine $i$'s wake. By convention, $A_{ii} = 0$ and $A_{ij} = 0$ if $x_j \leq x_i$ (turbine $j$ is upstream of turbine $i$).

Wake-induced mixing can affect both the velocity deficit and wake deflection.
To account for wake-induced mixing, the wake width of turbine $j$ is adjusted
to

$$ \sigma_{y}(x) = \int_{0}^x \sum_{i=0}^n k_i \ell_{[b_{i}, b_{i+1})}(x') + w_v \text{WIM}_j   dx' + \sigma_{y0} $$

where $w_v$ is the velocity deficit wake-induced mixing gain, which the
user can vary by setting `wim_gain_velocity` to represent different levels of
mixing caused by the turbines.

The wake deflection model is similarly adjusted to

$$ \delta = \frac{k_\text{def} C_T \alpha}{1 + w_d \text{WIM}_j}\operatorname{ln}\left(\frac{x/D - c}{x/D + c} + 2\right)$$

where $w_d$ is the wake-induced mixing gain for deflection, provided by the
user by setting `wim_gain_deflection`.

```{note}
The documentation previously had an erroneous form for the wake-induced mixing
term
$$ \text{WIM}_j = \sum_{i \in T^{\text{up}}(j)} \frac{A_{ij} a_i} {(x_j - x_i)/D_i} $$
This has been corrected to the form shown above, which is consistent with the implementation. The implementation has not changed; only the documentation has been corrected.
```

## Yaw added mixing

Yaw misalignment can also add turbulence to the wake. In the empirical Gaussian
model, this effect, referred to as "yaw-added wake recovery" in other models,
is activated by setting
`enable_yaw_added_recovery` to `true`. Yaw-added mixing is represented
by updating the wake-induced mixing term as follows:

$$ \text{WIM}_j = \sqrt{\sum_{i \neq j} \left(\frac{A_{ij} a_i (1 + g_\text{YAM} (1-\cos(\gamma_i)))}{\bar{d}_{ij}^2}\right)^2 + \left(a_j g_\text{YAM} (1-\cos(\gamma_j))\right)^2}$$

Note that the second term means that, unlike when `enable_yaw_added_recovery`
is `false`, a turbine may affect the recovery of its own wake by yawing.

## Mirror wakes

Mirror wakes are also enabled by default in the empirical model to model the
ground effect. Essentially, turbines are placed below the ground so that
the vertical expansion of their (mirror) wakes appears in the above-ground
flow some distance downstream, to model the reflection of the true turbine
wakes as they bounce off of the ground/sea surface.

## Added mixing by active wake control

As the name suggests, active wake control (AWC) aims to enhance mixing to the
wake of the controlled turbine. This effect is activated by setting
`enable_active_wake_mixing` to `true`, and `awc_modes` to `"helix"` (other AWC
strategies are yet to be implemented). The wake can then be controlled by
setting the amplitude of the AWC excitation using `awc_amplitudes` (see the
[AWC operation model](operation_models_user.ipynb#awc-model)).
The effect of AWC is represented by updating the
wake-induced mixing term as follows:

$$ \text{WIM}_j = \sqrt{\sum_{i \neq j} \left(\frac{A_{ij} a_i}{\bar{d}_{ij}^2}\right)^2 + \left(\frac{\beta_{j}^{p_\text{AWC}}}{d_\text{AWC}}\right)^2}$$

where $\beta_{j}$ is the AWC amplitude of turbine $j$, and the exponent $p_\text{AWC}$ and
denominator $d_\text{AWC}$ are tuning parameters that can be set in the `emgauss.yaml` file with
the fields `awc_wake_exp` and `awc_wake_denominator`, respectively.
Note that, in contrast to the yaw added mixing case, a turbine currently affects _only_ its own
wake by applying AWC.
