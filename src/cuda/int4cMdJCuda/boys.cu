/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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

#include <float.h>
#define EPS_FLOAT64     DBL_EPSILON
#define SQRTPIE4        .886226925452758013

__device__
static void eval_gamma_inc_fn(double *f, double t, int m, int sq_id, int block_size)
{
    if (t < EPS_FLOAT64) {
        f[sq_id] = 1.;
        for (int i = 1; i <= m; i++) {
            f[sq_id + i*block_size] = 1./(2*i+1);
        }
    } else if (m > 0 && t < m*.5+.5) {
        double bi = m + .5;
        double e = .5 * exp(-t);
        double x = e;
        double s = e;
        double tol = EPS_FLOAT64 * e;
        while (x > tol) {
            bi += 1.;
            x *= t / bi;
            s += x;
        }
        double b = m + 0.5;
        double fval = s / b;
        f[sq_id + m*block_size] = fval;
        for (int i = m-1; i >= 0; i--) {
            b -= 1.;
            fval = (e + t * fval) / b;
            f[sq_id + i*block_size] = fval;
        }
    } else {
        double tt = sqrt(t);
        double fval = SQRTPIE4 / tt * erf(tt);
        f[sq_id] = fval;
        if (m > 0) {
            double e = .5 * exp(-t);
            double b = 1. / t;
            double b1 = .5;
            for (int i = 1; i <= m; i++) {
                fval = b * (b1 * fval - e);
                f[sq_id + i*block_size] = fval;
                b1 += 1.;
            }
        }
    }
}


__device__
static void boys_fn(double *out, double theta, double rr, double omega,
                    double fac, int order, int thread_id, int threads)
{
    double theta_rr = theta * rr;
    if (omega == 0) {
        eval_gamma_inc_fn(out, theta_rr, order, thread_id, threads);
    } else if (omega > 0) {
        double theta_fac = omega * omega / (omega * omega + theta);
        eval_gamma_inc_fn(out, theta_fac*theta_rr, order, thread_id, threads);
        double scale = sqrt(theta_fac);
        for (int n = 0 ; n <= order; ++n) {
            out[thread_id+n*threads] *= scale;
            scale *= theta_fac;
        }
    } else { // omega < 0
        eval_gamma_inc_fn(out, theta_rr, order, thread_id, threads);
        double theta_fac = omega * omega / (omega * omega + theta);
        double *out1 = out + threads * (order+1);
        eval_gamma_inc_fn(out1, theta_fac*theta_rr, order, thread_id, threads);
        double scale = sqrt(theta_fac);
        for (int n = 0 ; n <= order; ++n) {
            out[thread_id+n*threads] -= scale * out1[thread_id+n*threads];
            scale *= theta_fac;
        }
    }
    out[thread_id] *= fac;
    double a2 = -2. * theta;
    for (int n = 1; n <= order; n++) {
        fac *= a2;
        out[thread_id+n*threads] *= fac;
    }
}
