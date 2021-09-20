/* libFLAC - Free Lossless Audio Codec library
 * Copyright (C) 2000-2009  Josh Coalson
 * Copyright (C) 2011-2016  Xiph.Org Foundation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * - Neither the name of the Xiph.org Foundation nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <math.h>
#include <stdlib.h>

#include "FLAC/assert.h"
#include "FLAC/format.h"
#include "share/compat.h"
#include "private/bitmath.h"
#include "private/lpc.h"
#include "private/macros.h"
#if !defined(NDEBUG) || defined FLAC__OVERFLOW_DETECT || defined FLAC__OVERFLOW_DETECT_VERBOSE
#include <stdio.h>
#endif

#define IRLS_MOVING_AVERAGE_WINDOW 128


/* OPT: #undef'ing this may improve the speed on some architectures */
#define FLAC__LPC_UNROLLED_FILTER_LOOPS

#ifndef FLAC__INTEGER_ONLY_LIBRARY

#if defined(_MSC_VER) && (_MSC_VER < 1800)
#include <float.h>
static inline long int lround(double x) {
	return (long)(x + _copysign(0.5, x));
}
#elif !defined(HAVE_LROUND) && defined(__GNUC__)
static inline long int lround(double x) {
	return (long)(x + __builtin_copysign(0.5, x));
}
/* If this fails, we are in the presence of a mid 90's compiler, move along... */
#endif

void FLAC__lpc_window_data(const FLAC__int32 in[], const FLAC__real window[], FLAC__real out[], uint32_t data_len)
{
	uint32_t i;
	for(i = 0; i < data_len; i++)
		out[i] = in[i] * window[i];
}

void FLAC__lpc_compute_autocorrelation(const FLAC__real data[], uint32_t data_len, uint32_t lag, double autoc[])
{
	/* a readable, but slower, version */
#if 0
	FLAC__real d;
	uint32_t i;

	FLAC__ASSERT(lag > 0);
	FLAC__ASSERT(lag <= data_len);

	/*
	 * Technically we should subtract the mean first like so:
	 *   for(i = 0; i < data_len; i++)
	 *     data[i] -= mean;
	 * but it appears not to make enough of a difference to matter, and
	 * most signals are already closely centered around zero
	 */
	while(lag--) {
		for(i = lag, d = 0.0; i < data_len; i++)
			d += data[i] * data[i - lag];
		autoc[lag] = d;
	}
#endif

	/*
	 * this version tends to run faster because of better data locality
	 * ('data_len' is usually much larger than 'lag')
	 */
	FLAC__real d;
	uint32_t sample, coeff;
	const uint32_t limit = data_len - lag;

	FLAC__ASSERT(lag > 0);
	FLAC__ASSERT(lag <= data_len);

	for(coeff = 0; coeff < lag; coeff++)
		autoc[coeff] = 0.0;
	for(sample = 0; sample <= limit; sample++) {
		d = data[sample];
		for(coeff = 0; coeff < lag; coeff++)
			autoc[coeff] += d * data[sample+coeff];
	}
	for(; sample < data_len; sample++) {
		d = data[sample];
		for(coeff = 0; coeff < data_len - sample; coeff++)
			autoc[coeff] += d * data[sample+coeff];
	}
}

void FLAC__lpc_compute_lp_coefficients(const double autoc[], uint32_t *max_order, FLAC__real lp_coeff[][FLAC__MAX_LPC_ORDER], double error[])
{
	uint32_t i, j;
	double r, err, lpc[FLAC__MAX_LPC_ORDER];

	FLAC__ASSERT(0 != max_order);
	FLAC__ASSERT(0 < *max_order);
	FLAC__ASSERT(*max_order <= FLAC__MAX_LPC_ORDER);
	FLAC__ASSERT(autoc[0] != 0.0);

	err = autoc[0];

	for(i = 0; i < *max_order; i++) {
		/* Sum up this iteration's reflection coefficient. */
		r = -autoc[i+1];
		for(j = 0; j < i; j++)
			r -= lpc[j] * autoc[i-j];
		r /= err;

		/* Update LPC coefficients and total error. */
		lpc[i]=r;
		for(j = 0; j < (i>>1); j++) {
			double tmp = lpc[j];
			lpc[j] += r * lpc[i-1-j];
			lpc[i-1-j] += r * tmp;
		}
		if(i & 1)
			lpc[j] += lpc[j] * r;

		err *= (1.0 - r * r);

		/* save this order */
		for(j = 0; j <= i; j++)
			lp_coeff[i][j] = (FLAC__real)(-lpc[j]); /* negate FIR filter coeff to get predictor coeff */
		error[i] = err;

		/* see SF bug https://sourceforge.net/p/flac/bugs/234/ */
		if(err == 0.0) {
			*max_order = i+1;
			return;
		}
	}
}
#ifdef ENABLE_ITERATIVELY_REWEIGHTED_LEAST_SQUARES
void FLAC__lpc_solve_symmetric_matrix(double A[][FLAC__MAX_LPC_ORDER], double b[], FLAC__real lp_coeff[][FLAC__MAX_LPC_ORDER], uint32_t order)
{
	int32_t i, j, k, l, order_signed;
	double x[FLAC__MAX_LPC_ORDER] = {0};

	FLAC__ASSERT(order <= FLAC__MAX_LPC_ORDER);

	order_signed = order;

    // Cholesky decomposition with the Cholesky–Crout algorithm
    // Overwrites symmetric o-by-o matrix A with lower triangular
    // o-by-o matrix L.
    for(j = 0; j < order_signed; j++){
        for(k = 0; k < j; k++){
            A[j][j] = A[j][j] - A[j][k]*A[j][k];
        }
        A[j][j] = sqrt(A[j][j]);
        for(k = j + 1; k < order_signed; k++){
            for(l = 0; l < j; l++){
                A[k][j] = A[k][j] - A[k][l] * A[j][l];
            }
            A[k][j] = A[k][j] / A[j][j];
        }
    }

    // Forward substitution of Lx = b, with x overwriting b
    for(j = 0; j < order_signed; j++){
        for(k = 0; k < j; k++){
            b[j] = b[j] - A[j][k] * b[k];
        }
        b[j] = b[j] / A[j][j];
    }

    // Up to here, steps are the same for all orders. The next
    // step has to be done for each order

    for(i = 0; i < order_signed; i++){
		for(j = 0; j <= i; j++)
			x[j] = b[j];
		// Backward substitution of L'x = b
		for(j = i; j >= 0; j--){
			for(k = i; k > j; k--){
				x[j] = x[j] - A[k][j] * x[k];
			}
			x[j] = x[j] / A[j][j];
			lp_coeff[i][j] = x[j];
		}
	}
}

FLAC__bool FLAC__lpc_weigh_data(const FLAC__int32 * flac_restrict data, FLAC__int32 * flac_restrict residual, double AWA[][FLAC__MAX_LPC_ORDER], double AWb[], uint32_t data_len, uint32_t order)
{
	double irls_moving_average_sum, irls_moving_average;
	double weight[FLAC__MAX_BLOCK_SIZE];
	double data_precast[FLAC__MAX_BLOCK_SIZE];

	// First, set AWA and AWb to 0
	for(uint32_t j = 0; j < order; j++){
		for(uint32_t k = 0; k <= j; k++){
			AWA[j][k] = 0.0;
		}
		AWb[j] = 0.0;
	}

	// Now, cast all data to double
	for(int i = -order; i < (int)data_len; i++)
		data_precast[i] = data[i];

	// We need a moving average to set the weighting cut-offs.
	// With this moving average, the rice parameter can be guessed
	// This needs a headstart. We need to skip the first "order" number
	// of samples as these contain invalid data

	irls_moving_average_sum = 0.0;
	for(uint32_t i = 0; i < data_len && i < (IRLS_MOVING_AVERAGE_WINDOW); i++){
		irls_moving_average_sum += abs(residual[i]);
	}


	// As the weight is the inverse of the residual
	// we're reusing the residual as the weighing variable
	// to speed things up

	for(uint32_t i = 0; i < data_len; i++){
		residual[i] = abs(residual[i]);
		irls_moving_average = irls_moving_average_sum/IRLS_MOVING_AVERAGE_WINDOW;
		if(residual[i] < 2)
			// This is a cap for very small residuals, so they don't get
			// too much attention
			weight[i] = 0.5/(irls_moving_average);
//		else if(residual[i] > (6*irls_moving_average))
			// Reducing large errors (compared to the moving average)
			// is usually not possible (in case of outliers) or sacrifices
			// the fit on other samples. To this end, residuals larger than
			// 2x the moving average get smaller weights. The multiplication
			// by the moving average*2 is to make the weighting continuous
//			weight[i] = 1.0/(irls_moving_average/6)/(residual[i]*residual[i]);
		else
			// The weight of a sample is the inverse of the moving average
			// times the inverse of the residual. By taking the inverse
			// of the residual
			weight[i] = 1.0/(irls_moving_average)/residual[i];

		// Update moving average when current sample is at least half a
		// window length away from beginning or end
		if(i >= IRLS_MOVING_AVERAGE_WINDOW/2 && (i+IRLS_MOVING_AVERAGE_WINDOW/2) < data_len){
			irls_moving_average_sum += abs(residual[i+IRLS_MOVING_AVERAGE_WINDOW/2]);
			irls_moving_average_sum -= residual[i-IRLS_MOVING_AVERAGE_WINDOW/2];
		}
	}

	// This loop runs over samples instead of over orders
	// because of data locality
	for(int i = 0; i < (int)data_len; i++){
		for(int j = 0; j < (int)order; j++){
			for(int k = 0; k <= j; k++){
				AWA[j][k] += weight[i]*data_precast[i-j-1]*data_precast[i-k-1];
			}
			AWb[j] += weight[i]*data_precast[i-j-1]*data_precast[i];
		}
	}

	for(uint32_t i = 0; i < order; i++){
        if(AWA[i][i] < 1){
			return false;
 		}
        if(AWb[i] != AWb[i]){
			return false;
		}
	}
	return true;
}

FLAC__bool FLAC__lpc_iterate_weighted_least_squares(const FLAC__int32 * flac_restrict data, FLAC__real lp_coeff[][FLAC__MAX_LPC_ORDER], uint32_t data_len, uint32_t max_order, uint32_t iterations, void (*local_lpc_compute_residual_from_qlp_coefficients_64bit)(const FLAC__int32[], uint32_t, const FLAC__int32[], uint32_t order, int lp_quantization, FLAC__int32[]), FLAC__bool reuse_lpcoeff)
{
	double AWA[FLAC__MAX_LPC_ORDER][FLAC__MAX_LPC_ORDER] = {0};
	double AWb[FLAC__MAX_LPC_ORDER] = {0};
	FLAC__int32 residual[FLAC__MAX_BLOCK_SIZE];
	uint32_t i,j;
	int quantization;
	FLAC__int32 qlp_coeff[FLAC__MAX_LPC_ORDER];

	if(data_len <= max_order)
		return false;
	for(i = 0; i < flac_max(iterations,(uint32_t)1); i++){
		if(i == 0 && !reuse_lpcoeff){
			// For i == 0, we start with no weighting, except when reuse_lpcoeff is set
			for(j = 0; j < data_len; j++){
				residual[j] = 1;
			}
		}else{
			// Copy predictor from lp_coeff
			FLAC__lpc_quantize_coefficients(lp_coeff[max_order-1], max_order, 16, qlp_coeff, &quantization);
			local_lpc_compute_residual_from_qlp_coefficients_64bit(data+max_order, (data_len-max_order), qlp_coeff, max_order, quantization, residual);
		}
		if(!FLAC__lpc_weigh_data(data+max_order,residual,AWA,AWb,(data_len-max_order),max_order))
			return false;
		FLAC__lpc_solve_symmetric_matrix(AWA,AWb,lp_coeff,max_order);

	}

    return true;
}
#endif /* end of ifdef ENABLE_ITERATIVELY_REWEIGHTED_LEAST_SQUARES */

int FLAC__lpc_quantize_coefficients(const FLAC__real lp_coeff[], uint32_t order, uint32_t precision, FLAC__int32 qlp_coeff[], int *shift)
{
	uint32_t i;
	double cmax;
	FLAC__int32 qmax, qmin;

	FLAC__ASSERT(precision > 0);
	FLAC__ASSERT(precision >= FLAC__MIN_QLP_COEFF_PRECISION);

	/* drop one bit for the sign; from here on out we consider only |lp_coeff[i]| */
	precision--;
	qmax = 1 << precision;
	qmin = -qmax;
	qmax--;

	/* calc cmax = max( |lp_coeff[i]| ) */
	cmax = 0.0;
	for(i = 0; i < order; i++) {
		const double d = fabs(lp_coeff[i]);
		if(d > cmax)
			cmax = d;
	}

	if(cmax <= 0.0) {
		/* => coefficients are all 0, which means our constant-detect didn't work */
		return 2;
	}
	else {
		const int max_shiftlimit = (1 << (FLAC__SUBFRAME_LPC_QLP_SHIFT_LEN-1)) - 1;
		const int min_shiftlimit = -max_shiftlimit - 1;
		int log2cmax;

		(void)frexp(cmax, &log2cmax);
		log2cmax--;
		*shift = (int)precision - log2cmax - 1;

		if(*shift > max_shiftlimit)
			*shift = max_shiftlimit;
		else if(*shift < min_shiftlimit)
			return 1;
	}

	if(*shift >= 0) {
		double error = 0.0;
		FLAC__int32 q;
		for(i = 0; i < order; i++) {
			error += lp_coeff[i] * (1 << *shift);
			q = lround(error);

#ifdef FLAC__OVERFLOW_DETECT
			if(q > qmax+1) /* we expect q==qmax+1 occasionally due to rounding */
				fprintf(stderr,"FLAC__lpc_quantize_coefficients: quantizer overflow: q>qmax %d>%d shift=%d cmax=%f precision=%u lpc[%u]=%f\n",q,qmax,*shift,cmax,precision+1,i,lp_coeff[i]);
			else if(q < qmin)
				fprintf(stderr,"FLAC__lpc_quantize_coefficients: quantizer overflow: q<qmin %d<%d shift=%d cmax=%f precision=%u lpc[%u]=%f\n",q,qmin,*shift,cmax,precision+1,i,lp_coeff[i]);
#endif
			if(q > qmax)
				q = qmax;
			else if(q < qmin)
				q = qmin;
			error -= q;
			qlp_coeff[i] = q;
		}
	}
	/* negative shift is very rare but due to design flaw, negative shift is
	 * not allowed in the decoder, so it must be handled specially by scaling
	 * down coeffs
	 */
	else {
		const int nshift = -(*shift);
		double error = 0.0;
		FLAC__int32 q;
#ifndef NDEBUG
		fprintf(stderr,"FLAC__lpc_quantize_coefficients: negative shift=%d order=%u cmax=%f\n", *shift, order, cmax);
#endif
		for(i = 0; i < order; i++) {
			error += lp_coeff[i] / (1 << nshift);
			q = lround(error);
#ifdef FLAC__OVERFLOW_DETECT
			if(q > qmax+1) /* we expect q==qmax+1 occasionally due to rounding */
				fprintf(stderr,"FLAC__lpc_quantize_coefficients: quantizer overflow: q>qmax %d>%d shift=%d cmax=%f precision=%u lpc[%u]=%f\n",q,qmax,*shift,cmax,precision+1,i,lp_coeff[i]);
			else if(q < qmin)
				fprintf(stderr,"FLAC__lpc_quantize_coefficients: quantizer overflow: q<qmin %d<%d shift=%d cmax=%f precision=%u lpc[%u]=%f\n",q,qmin,*shift,cmax,precision+1,i,lp_coeff[i]);
#endif
			if(q > qmax)
				q = qmax;
			else if(q < qmin)
				q = qmin;
			error -= q;
			qlp_coeff[i] = q;
		}
		*shift = 0;
	}

	return 0;
}

#if defined(_MSC_VER)
// silence MSVC warnings about __restrict modifier
#pragma warning ( disable : 4028 )
#endif

void FLAC__lpc_compute_residual_from_qlp_coefficients(const FLAC__int32 * flac_restrict data, uint32_t data_len, const FLAC__int32 * flac_restrict qlp_coeff, uint32_t order, int lp_quantization, FLAC__int32 * flac_restrict residual)
#if defined(FLAC__OVERFLOW_DETECT) || !defined(FLAC__LPC_UNROLLED_FILTER_LOOPS)
{
	FLAC__int64 sumo;
	uint32_t i, j;
	FLAC__int32 sum;
	const FLAC__int32 *history;

#ifdef FLAC__OVERFLOW_DETECT_VERBOSE
	fprintf(stderr,"FLAC__lpc_compute_residual_from_qlp_coefficients: data_len=%d, order=%u, lpq=%d",data_len,order,lp_quantization);
	for(i=0;i<order;i++)
		fprintf(stderr,", q[%u]=%d",i,qlp_coeff[i]);
	fprintf(stderr,"\n");
#endif
	FLAC__ASSERT(order > 0);

	for(i = 0; i < data_len; i++) {
		sumo = 0;
		sum = 0;
		history = data;
		for(j = 0; j < order; j++) {
			sum += qlp_coeff[j] * (*(--history));
			sumo += (FLAC__int64)qlp_coeff[j] * (FLAC__int64)(*history);
			if(sumo > 2147483647ll || sumo < -2147483648ll)
				fprintf(stderr,"FLAC__lpc_compute_residual_from_qlp_coefficients: OVERFLOW, i=%u, j=%u, c=%d, d=%d, sumo=%" PRId64 "\n",i,j,qlp_coeff[j],*history,sumo);
		}
		*(residual++) = *(data++) - (sum >> lp_quantization);
	}

	/* Here's a slower but clearer version:
	for(i = 0; i < data_len; i++) {
		sum = 0;
		for(j = 0; j < order; j++)
			sum += qlp_coeff[j] * data[i-j-1];
		residual[i] = data[i] - (sum >> lp_quantization);
	}
	*/
}
#else /* fully unrolled version for normal use */
{
	int i;
	FLAC__int32 sum;

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	/*
	 * We do unique versions up to 12th order since that's the subset limit.
	 * Also they are roughly ordered to match frequency of occurrence to
	 * minimize branching.
	 */
	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[11] * data[i-12];
						sum += qlp_coeff[10] * data[i-11];
						sum += qlp_coeff[9] * data[i-10];
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
				else { /* order == 11 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[10] * data[i-11];
						sum += qlp_coeff[9] * data[i-10];
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 10) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[9] * data[i-10];
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
				else { /* order == 9 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
			}
		}
		else if(order > 4) {
			if(order > 6) {
				if(order == 8) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
				else { /* order == 7 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 6) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
				else { /* order == 5 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
			}
		}
		else {
			if(order > 2) {
				if(order == 4) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
				else { /* order == 3 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 2) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						residual[i] = data[i] - (sum >> lp_quantization);
					}
				}
				else { /* order == 1 */
					for(i = 0; i < (int)data_len; i++)
						residual[i] = data[i] - ((qlp_coeff[0] * data[i-1]) >> lp_quantization);
				}
			}
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * data[i-13];
				         sum += qlp_coeff[11] * data[i-12];
				         sum += qlp_coeff[10] * data[i-11];
				         sum += qlp_coeff[ 9] * data[i-10];
				         sum += qlp_coeff[ 8] * data[i- 9];
				         sum += qlp_coeff[ 7] * data[i- 8];
				         sum += qlp_coeff[ 6] * data[i- 7];
				         sum += qlp_coeff[ 5] * data[i- 6];
				         sum += qlp_coeff[ 4] * data[i- 5];
				         sum += qlp_coeff[ 3] * data[i- 4];
				         sum += qlp_coeff[ 2] * data[i- 3];
				         sum += qlp_coeff[ 1] * data[i- 2];
				         sum += qlp_coeff[ 0] * data[i- 1];
			}
			residual[i] = data[i] - (sum >> lp_quantization);
		}
	}
}
#endif

void FLAC__lpc_compute_residual_from_qlp_coefficients_wide(const FLAC__int32 * flac_restrict data, uint32_t data_len, const FLAC__int32 * flac_restrict qlp_coeff, uint32_t order, int lp_quantization, FLAC__int32 * flac_restrict residual)
#if defined(FLAC__OVERFLOW_DETECT) || !defined(FLAC__LPC_UNROLLED_FILTER_LOOPS)
{
	uint32_t i, j;
	FLAC__int64 sum;
	const FLAC__int32 *history;

#ifdef FLAC__OVERFLOW_DETECT_VERBOSE
	fprintf(stderr,"FLAC__lpc_compute_residual_from_qlp_coefficients_wide: data_len=%d, order=%u, lpq=%d",data_len,order,lp_quantization);
	for(i=0;i<order;i++)
		fprintf(stderr,", q[%u]=%d",i,qlp_coeff[i]);
	fprintf(stderr,"\n");
#endif
	FLAC__ASSERT(order > 0);

	for(i = 0; i < data_len; i++) {
		sum = 0;
		history = data;
		for(j = 0; j < order; j++)
			sum += (FLAC__int64)qlp_coeff[j] * (FLAC__int64)(*(--history));
		if(FLAC__bitmath_silog2(sum >> lp_quantization) > 32) {
			fprintf(stderr,"FLAC__lpc_compute_residual_from_qlp_coefficients_wide: OVERFLOW, i=%u, sum=%" PRId64 "\n", i, (sum >> lp_quantization));
			break;
		}
		if(FLAC__bitmath_silog2((FLAC__int64)(*data) - (sum >> lp_quantization)) > 32) {
			fprintf(stderr,"FLAC__lpc_compute_residual_from_qlp_coefficients_wide: OVERFLOW, i=%u, data=%d, sum=%" PRId64 ", residual=%" PRId64 "\n", i, *data, (int64_t)(sum >> lp_quantization), ((FLAC__int64)(*data) - (sum >> lp_quantization)));
			break;
		}
		*(residual++) = *(data++) - (FLAC__int32)(sum >> lp_quantization);
	}
}
#else /* fully unrolled version for normal use */
{
	int i;
	FLAC__int64 sum;

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	/*
	 * We do unique versions up to 12th order since that's the subset limit.
	 * Also they are roughly ordered to match frequency of occurrence to
	 * minimize branching.
	 */
	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[11] * (FLAC__int64)data[i-12];
						sum += qlp_coeff[10] * (FLAC__int64)data[i-11];
						sum += qlp_coeff[9] * (FLAC__int64)data[i-10];
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
				else { /* order == 11 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[10] * (FLAC__int64)data[i-11];
						sum += qlp_coeff[9] * (FLAC__int64)data[i-10];
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 10) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[9] * (FLAC__int64)data[i-10];
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
				else { /* order == 9 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
			}
		}
		else if(order > 4) {
			if(order > 6) {
				if(order == 8) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
				else { /* order == 7 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 6) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
				else { /* order == 5 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
			}
		}
		else {
			if(order > 2) {
				if(order == 4) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
				else { /* order == 3 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 2) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
					}
				}
				else { /* order == 1 */
					for(i = 0; i < (int)data_len; i++)
						residual[i] = data[i] - (FLAC__int32)((qlp_coeff[0] * (FLAC__int64)data[i-1]) >> lp_quantization);
				}
			}
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * (FLAC__int64)data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * (FLAC__int64)data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * (FLAC__int64)data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * (FLAC__int64)data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * (FLAC__int64)data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * (FLAC__int64)data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * (FLAC__int64)data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * (FLAC__int64)data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * (FLAC__int64)data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * (FLAC__int64)data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * (FLAC__int64)data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * (FLAC__int64)data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * (FLAC__int64)data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * (FLAC__int64)data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * (FLAC__int64)data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * (FLAC__int64)data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * (FLAC__int64)data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * (FLAC__int64)data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * (FLAC__int64)data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * (FLAC__int64)data[i-13];
				         sum += qlp_coeff[11] * (FLAC__int64)data[i-12];
				         sum += qlp_coeff[10] * (FLAC__int64)data[i-11];
				         sum += qlp_coeff[ 9] * (FLAC__int64)data[i-10];
				         sum += qlp_coeff[ 8] * (FLAC__int64)data[i- 9];
				         sum += qlp_coeff[ 7] * (FLAC__int64)data[i- 8];
				         sum += qlp_coeff[ 6] * (FLAC__int64)data[i- 7];
				         sum += qlp_coeff[ 5] * (FLAC__int64)data[i- 6];
				         sum += qlp_coeff[ 4] * (FLAC__int64)data[i- 5];
				         sum += qlp_coeff[ 3] * (FLAC__int64)data[i- 4];
				         sum += qlp_coeff[ 2] * (FLAC__int64)data[i- 3];
				         sum += qlp_coeff[ 1] * (FLAC__int64)data[i- 2];
				         sum += qlp_coeff[ 0] * (FLAC__int64)data[i- 1];
			}
			residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
		}
	}
}
#endif

#endif /* !defined FLAC__INTEGER_ONLY_LIBRARY */

void FLAC__lpc_restore_signal(const FLAC__int32 * flac_restrict residual, uint32_t data_len, const FLAC__int32 * flac_restrict qlp_coeff, uint32_t order, int lp_quantization, FLAC__int32 * flac_restrict data)
#if defined(FLAC__OVERFLOW_DETECT) || !defined(FLAC__LPC_UNROLLED_FILTER_LOOPS)
{
	FLAC__int64 sumo;
	uint32_t i, j;
	FLAC__int32 sum;
	const FLAC__int32 *r = residual, *history;

#ifdef FLAC__OVERFLOW_DETECT_VERBOSE
	fprintf(stderr,"FLAC__lpc_restore_signal: data_len=%d, order=%u, lpq=%d",data_len,order,lp_quantization);
	for(i=0;i<order;i++)
		fprintf(stderr,", q[%u]=%d",i,qlp_coeff[i]);
	fprintf(stderr,"\n");
#endif
	FLAC__ASSERT(order > 0);

	for(i = 0; i < data_len; i++) {
		sumo = 0;
		sum = 0;
		history = data;
		for(j = 0; j < order; j++) {
			sum += qlp_coeff[j] * (*(--history));
			sumo += (FLAC__int64)qlp_coeff[j] * (FLAC__int64)(*history);
			if(sumo > 2147483647ll || sumo < -2147483648ll)
				fprintf(stderr,"FLAC__lpc_restore_signal: OVERFLOW, i=%u, j=%u, c=%d, d=%d, sumo=%" PRId64 "\n",i,j,qlp_coeff[j],*history,sumo);
		}
		*(data++) = *(r++) + (sum >> lp_quantization);
	}

	/* Here's a slower but clearer version:
	for(i = 0; i < data_len; i++) {
		sum = 0;
		for(j = 0; j < order; j++)
			sum += qlp_coeff[j] * data[i-j-1];
		data[i] = residual[i] + (sum >> lp_quantization);
	}
	*/
}
#else /* fully unrolled version for normal use */
{
	int i;
	FLAC__int32 sum;

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	/*
	 * We do unique versions up to 12th order since that's the subset limit.
	 * Also they are roughly ordered to match frequency of occurrence to
	 * minimize branching.
	 */
	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[11] * data[i-12];
						sum += qlp_coeff[10] * data[i-11];
						sum += qlp_coeff[9] * data[i-10];
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
				else { /* order == 11 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[10] * data[i-11];
						sum += qlp_coeff[9] * data[i-10];
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 10) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[9] * data[i-10];
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
				else { /* order == 9 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[8] * data[i-9];
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
			}
		}
		else if(order > 4) {
			if(order > 6) {
				if(order == 8) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[7] * data[i-8];
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
				else { /* order == 7 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[6] * data[i-7];
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 6) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[5] * data[i-6];
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
				else { /* order == 5 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[4] * data[i-5];
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
			}
		}
		else {
			if(order > 2) {
				if(order == 4) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[3] * data[i-4];
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
				else { /* order == 3 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[2] * data[i-3];
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
			}
			else {
				if(order == 2) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[1] * data[i-2];
						sum += qlp_coeff[0] * data[i-1];
						data[i] = residual[i] + (sum >> lp_quantization);
					}
				}
				else { /* order == 1 */
					for(i = 0; i < (int)data_len; i++)
						data[i] = residual[i] + ((qlp_coeff[0] * data[i-1]) >> lp_quantization);
				}
			}
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * data[i-13];
				         sum += qlp_coeff[11] * data[i-12];
				         sum += qlp_coeff[10] * data[i-11];
				         sum += qlp_coeff[ 9] * data[i-10];
				         sum += qlp_coeff[ 8] * data[i- 9];
				         sum += qlp_coeff[ 7] * data[i- 8];
				         sum += qlp_coeff[ 6] * data[i- 7];
				         sum += qlp_coeff[ 5] * data[i- 6];
				         sum += qlp_coeff[ 4] * data[i- 5];
				         sum += qlp_coeff[ 3] * data[i- 4];
				         sum += qlp_coeff[ 2] * data[i- 3];
				         sum += qlp_coeff[ 1] * data[i- 2];
				         sum += qlp_coeff[ 0] * data[i- 1];
			}
			data[i] = residual[i] + (sum >> lp_quantization);
		}
	}
}
#endif

void FLAC__lpc_restore_signal_wide(const FLAC__int32 * flac_restrict residual, uint32_t data_len, const FLAC__int32 * flac_restrict qlp_coeff, uint32_t order, int lp_quantization, FLAC__int32 * flac_restrict data)
#if defined(FLAC__OVERFLOW_DETECT) || !defined(FLAC__LPC_UNROLLED_FILTER_LOOPS)
{
	uint32_t i, j;
	FLAC__int64 sum;
	const FLAC__int32 *r = residual, *history;

#ifdef FLAC__OVERFLOW_DETECT_VERBOSE
	fprintf(stderr,"FLAC__lpc_restore_signal_wide: data_len=%d, order=%u, lpq=%d",data_len,order,lp_quantization);
	for(i=0;i<order;i++)
		fprintf(stderr,", q[%u]=%d",i,qlp_coeff[i]);
	fprintf(stderr,"\n");
#endif
	FLAC__ASSERT(order > 0);

	for(i = 0; i < data_len; i++) {
		sum = 0;
		history = data;
		for(j = 0; j < order; j++)
			sum += (FLAC__int64)qlp_coeff[j] * (FLAC__int64)(*(--history));
		if(FLAC__bitmath_silog2(sum >> lp_quantization) > 32) {
			fprintf(stderr,"FLAC__lpc_restore_signal_wide: OVERFLOW, i=%u, sum=%" PRId64 "\n", i, (sum >> lp_quantization));
			break;
		}
		if(FLAC__bitmath_silog2((FLAC__int64)(*r) + (sum >> lp_quantization)) > 32) {
			fprintf(stderr,"FLAC__lpc_restore_signal_wide: OVERFLOW, i=%u, residual=%d, sum=%" PRId64 ", data=%" PRId64 "\n", i, *r, (sum >> lp_quantization), ((FLAC__int64)(*r) + (sum >> lp_quantization)));
			break;
		}
		*(data++) = *(r++) + (FLAC__int32)(sum >> lp_quantization);
	}
}
#else /* fully unrolled version for normal use */
{
	int i;
	FLAC__int64 sum;

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	/*
	 * We do unique versions up to 12th order since that's the subset limit.
	 * Also they are roughly ordered to match frequency of occurrence to
	 * minimize branching.
	 */
	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[11] * (FLAC__int64)data[i-12];
						sum += qlp_coeff[10] * (FLAC__int64)data[i-11];
						sum += qlp_coeff[9] * (FLAC__int64)data[i-10];
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
				else { /* order == 11 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[10] * (FLAC__int64)data[i-11];
						sum += qlp_coeff[9] * (FLAC__int64)data[i-10];
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
			}
			else {
				if(order == 10) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[9] * (FLAC__int64)data[i-10];
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
				else { /* order == 9 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[8] * (FLAC__int64)data[i-9];
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
			}
		}
		else if(order > 4) {
			if(order > 6) {
				if(order == 8) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[7] * (FLAC__int64)data[i-8];
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
				else { /* order == 7 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[6] * (FLAC__int64)data[i-7];
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
			}
			else {
				if(order == 6) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[5] * (FLAC__int64)data[i-6];
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
				else { /* order == 5 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[4] * (FLAC__int64)data[i-5];
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
			}
		}
		else {
			if(order > 2) {
				if(order == 4) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[3] * (FLAC__int64)data[i-4];
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
				else { /* order == 3 */
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[2] * (FLAC__int64)data[i-3];
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
			}
			else {
				if(order == 2) {
					for(i = 0; i < (int)data_len; i++) {
						sum = 0;
						sum += qlp_coeff[1] * (FLAC__int64)data[i-2];
						sum += qlp_coeff[0] * (FLAC__int64)data[i-1];
						data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
					}
				}
				else { /* order == 1 */
					for(i = 0; i < (int)data_len; i++)
						data[i] = residual[i] + (FLAC__int32)((qlp_coeff[0] * (FLAC__int64)data[i-1]) >> lp_quantization);
				}
			}
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * (FLAC__int64)data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * (FLAC__int64)data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * (FLAC__int64)data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * (FLAC__int64)data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * (FLAC__int64)data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * (FLAC__int64)data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * (FLAC__int64)data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * (FLAC__int64)data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * (FLAC__int64)data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * (FLAC__int64)data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * (FLAC__int64)data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * (FLAC__int64)data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * (FLAC__int64)data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * (FLAC__int64)data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * (FLAC__int64)data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * (FLAC__int64)data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * (FLAC__int64)data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * (FLAC__int64)data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * (FLAC__int64)data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * (FLAC__int64)data[i-13];
				         sum += qlp_coeff[11] * (FLAC__int64)data[i-12];
				         sum += qlp_coeff[10] * (FLAC__int64)data[i-11];
				         sum += qlp_coeff[ 9] * (FLAC__int64)data[i-10];
				         sum += qlp_coeff[ 8] * (FLAC__int64)data[i- 9];
				         sum += qlp_coeff[ 7] * (FLAC__int64)data[i- 8];
				         sum += qlp_coeff[ 6] * (FLAC__int64)data[i- 7];
				         sum += qlp_coeff[ 5] * (FLAC__int64)data[i- 6];
				         sum += qlp_coeff[ 4] * (FLAC__int64)data[i- 5];
				         sum += qlp_coeff[ 3] * (FLAC__int64)data[i- 4];
				         sum += qlp_coeff[ 2] * (FLAC__int64)data[i- 3];
				         sum += qlp_coeff[ 1] * (FLAC__int64)data[i- 2];
				         sum += qlp_coeff[ 0] * (FLAC__int64)data[i- 1];
			}
			data[i] = (FLAC__int32) (residual[i] + (sum >> lp_quantization));
		}
	}
}
#endif

#if defined(_MSC_VER)
#pragma warning ( default : 4028 )
#endif

#ifndef FLAC__INTEGER_ONLY_LIBRARY

double FLAC__lpc_compute_expected_bits_per_residual_sample(double lpc_error, uint32_t total_samples)
{
	double error_scale;

	FLAC__ASSERT(total_samples > 0);

	error_scale = 0.5 / (double)total_samples;

	return FLAC__lpc_compute_expected_bits_per_residual_sample_with_error_scale(lpc_error, error_scale);
}

double FLAC__lpc_compute_expected_bits_per_residual_sample_with_error_scale(double lpc_error, double error_scale)
{
	if(lpc_error > 0.0) {
		double bps = (double)0.5 * log(error_scale * lpc_error) / M_LN2;
		if(bps >= 0.0)
			return bps;
		else
			return 0.0;
	}
	else if(lpc_error < 0.0) { /* error should not be negative but can happen due to inadequate floating-point resolution */
		return 1e32;
	}
	else {
		return 0.0;
	}
}
#ifdef ENABLE_ITERATIVELY_REWEIGHTED_LEAST_SQUARES
double FLAC__lpc_compute_expected_bits_per_residual_sample_with_abs_error(double lpc_error)
{
	if(lpc_error > 0.0) {
		double bps = log(M_LN2 * lpc_error) / M_LN2;
		if(bps >= 0.0)
			return bps;
		else
			return 0.0;
	}
	else if(lpc_error < 0.0) { /* error should not be negative but can happen due to inadequate floating-point resolution */
		return 1e32;
	}
	else {
		return 0.0;
	}
}
#endif /* end of ifdef ENABLE_ITERATIVELY_REWEIGHTED_LEAST_SQUARES */

uint32_t FLAC__lpc_compute_best_order(const double lpc_error[], uint32_t max_order, uint32_t total_samples, uint32_t overhead_bits_per_order)
{
	uint32_t order, indx, best_index; /* 'index' the index into lpc_error; index==order-1 since lpc_error[0] is for order==1, lpc_error[1] is for order==2, etc */
	double bits, best_bits, error_scale;

	FLAC__ASSERT(max_order > 0);
	FLAC__ASSERT(total_samples > 0);

	error_scale = 0.5 / (double)total_samples;

	best_index = 0;
	best_bits = (uint32_t)(-1);

	for(indx = 0, order = 1; indx < max_order; indx++, order++) {
		bits = FLAC__lpc_compute_expected_bits_per_residual_sample_with_error_scale(lpc_error[indx], error_scale) * (double)(total_samples - order) + (double)(order * overhead_bits_per_order);
		if(bits < best_bits) {
			best_index = indx;
			best_bits = bits;
		}
	}

	return best_index+1; /* +1 since indx of lpc_error[] is order-1 */
}

#endif /* !defined FLAC__INTEGER_ONLY_LIBRARY */
