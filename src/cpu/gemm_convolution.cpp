/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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
*******************************************************************************/

#include <common/primitive_attr.hpp>
#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "gemm_convolution.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

namespace {
struct im_pos_t {
    im_pos_t() : n{ 0 }, g{ 0 }, od{ 0 }, sp{ 0 }, ic{ 0 }, oc{ 0 } {}
    int n, g, od, sp, ic, oc;
    bool do_im2col(const im_pos_t &prev) const {
        return true
                && (n != prev.n || g != prev.g || od != prev.od || sp != prev.sp
                           || ic != prev.ic);
    }
};
} // namespace

template <cpu_isa_t isa>
jit_pp_kernel_t<isa>::jit_pp_kernel_t(
        const cpu_convolution_fwd_pd_t *pd)
        : ker_(nullptr)
        , do_bias_(pd->with_bias())
        , idx_compute_vreg_start_(0)
        , idx_compute_vreg_max_(isa == avx512_common ? 31 : 15) {
    using namespace types;
    using namespace Xbyak;

    if (utils::one_of(isa, avx2, sse42)) {
        idx_compute_vreg_start_ += 1;   //  Vmm(0) - for masks
    }
    post_ops_ = pd->attr()->post_ops_;
    bool only_eltwise = true;
    for (int i = 0; i < post_ops_.len_; i++) {
        auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors_.push_back(new jit_uni_eltwise_injector_f32<isa>(
                    this, post_op.eltwise, true, eltwise_reserved_1_, eltwise_reserved_2_));
        } else if (post_op.is_depthwise()) {
            only_eltwise = false;
            depthwise_injectors_.push_back(new jit_uni_depthwise_injector_f32<isa>(
                    this, post_op.depthwise.alg, depthwise_reserved_2_));
        } else {
            only_eltwise = false;
        }
    }
    if (post_ops_.len_ > 0 && !only_eltwise) {
        vreg_d_weights = Vmm(idx_compute_vreg_max_--);
        vreg_d_bias = Vmm(idx_compute_vreg_max_--);
    }
    if (utils::one_of(isa, avx2, sse42))
        vreg_zero = Vmm(idx_compute_vreg_start_++);

    generate();
}

ref_pp_kernel_t::ref_pp_kernel_t(
        const cpu_convolution_fwd_pd_t *pd)
        : do_bias_(pd->with_bias()) {
    post_ops_ = pd->attr()->post_ops_;
    for (int i = 0; i < post_ops_.len_; i++) {
        auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            ref_eltwise_impls_.push_back(new ref_eltwise_scalar_fwd_t(
                    post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
        } else if (post_op.is_depthwise()) {
            ref_depthwise_impls_.push_back(new ref_depthwise_scalar_fwd_t(
                    post_op.depthwise.alg));
        }
    }

    use_fast_relu = false;
    if (post_ops_.len_ == 1 && post_ops_.entry_[0].is_relu(true, false)) {
        use_fast_relu = true;
        fast_relu_ns = post_ops_.entry_[0].eltwise.alpha;
    } else if (post_ops_.len_ == 2 && post_ops_.entry_[0].is_sum() && post_ops_.entry_[1].is_relu(true, false)) {
        use_fast_relu = true;
        fast_relu_ns = post_ops_.entry_[1].eltwise.alpha;
    }
}

template<cpu_isa_t isa>
void jit_pp_kernel_t<isa>::generate()
{
    using namespace Xbyak;
    using namespace utils;
    using namespace round_mode;

    preamble();

    using ker_args_ = ker_args;
#define PARAM_OFF(x) offsetof(ker_args_, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
#undef PARAM_OFF

    if (utils::one_of(isa, avx2, sse42)) {
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);
        mov(reg_table, l_table);
    }

    auto apply_post_ops = [&]() {
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        auto vreg_dst_ = vreg_dst(0);
        for (int i = 0; i < post_ops_.len_; i++) {
            auto& post_op = post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors_[eltwise_inj_idx]->compute_vector(vreg_dst_.getIdx());
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));
                lea(reg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                lea(reg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                depthwise_injectors_[depthwise_inj_idx]->compute_vector_range(vreg_dst_.getIdx(), vreg_dst_.getIdx() + 1,
                        reg_d_weights, reg_d_bias, true);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = true;

                if (post_op.quantization.crop_low_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_));
                    uni_vpbroadcastd(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_));
                    uni_vpbroadcastd(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.crop_high_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_));
                    uni_vpbroadcastd(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_));
                    uni_vpbroadcastd(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vmaxps(vreg_dst_, vreg_dst_, vreg_d_weights);
                uni_vminps(vreg_dst_, vreg_dst_, vreg_d_bias);

                if (post_op.quantization.input_scale_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_));
                    uni_vpbroadcastd(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_));
                    uni_vpbroadcastd(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.input_shift_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_));
                    uni_vpbroadcastd(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_));
                    uni_vpbroadcastd(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vfmadd213ps(vreg_dst_, vreg_d_weights, vreg_d_bias);

                if (do_rounding)
                    uni_vroundps(vreg_dst_, vreg_dst_, 0);

                if (do_dequantization) {
                    if (post_op.quantization.output_scale_data->count_ != 1) {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_));
                        uni_vpbroadcastd(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_));
                        uni_vpbroadcastd(vreg_d_weights, ptr[reg_d_weights]);
                    }

                    if (post_op.quantization.output_shift_data->count_ != 1) {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_));
                        uni_vpbroadcastd(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_));
                        uni_vpbroadcastd(vreg_d_bias, ptr[reg_d_bias]);
                    }

                    uni_vfmadd213ps(vreg_dst_, vreg_d_weights, vreg_d_bias);
                }
            }
        }
    };

    // Load accumulated value, convert to float, apply bias (if any), scaling,
    // and eltwise (if any); then convert to destination type and store
    auto compute = [&](bool apply_mask) {
        auto dst_addr = ptr[reg_dst];
        auto vreg_dst_ = vreg_dst(0);
        if (isa == avx512_common) {
            if (apply_mask)
                vreg_dst_ = vreg_dst_ | kreg_rem_mask;
            uni_vmovups(vreg_dst_, dst_addr);
        } else {
            if (apply_mask) {
                if (isa != sse42) {
                    uni_vblendvps(vreg_dst_, vreg_zero, dst_addr, vreg_mask);
                } else {
                    uni_vmovups(vreg_dst_, dst_addr);
                }
            } else {
                uni_vmovups(vreg_dst_, dst_addr);
            }
        }

        if (do_bias_) {
            auto vreg_bias_ = vreg_bias(0);
            if (isa == avx512_common && apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask;

            uni_vpbroadcastd(vreg_bias_, ptr[reg_bias]);
            uni_vaddps(vreg_dst_, vreg_dst_, vreg_bias_);
        }

        apply_post_ops();

        if (isa == avx512_common) {
            uni_vmovups(dst_addr, vreg_dst_);
        } else {
            if (apply_mask) {
                if (isa != sse42) {
                    vmaskmovps(dst_addr, vreg_mask, vreg_dst_);
                } else {
                    lea(reg_ptr_maskmovdqu_dst, dst_addr);
                    maskmovdqu(vreg_dst_, vreg_mask);
                }
            } else {
                uni_vmovups(dst_addr, vreg_dst_);
            }
        }
    };

    Label loop_end;
    {
        cmp(reg_len, 0);
        je(loop_end, T_NEAR);

        Label loop, loop_tail;
        cmp(reg_len, vlen);
        jl(loop_tail, T_NEAR);
        L(loop); {
            compute(false);
            sub(reg_len, vlen);
            add(reg_dst, vlen * sizeof(float));
            cmp(reg_len, vlen);
            jge(loop, T_NEAR);
        }

        L(loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        if (isa == avx512_common) {
            mov(reg_rem_mask, 1);
            shl(reg_rem_mask, cl); // reg_tmp == rcx and reg_tail < vlen == 16
            sub(reg_rem_mask, 1);
            jz(loop_end, T_NEAR);
            kmovq(kreg_rem_mask, reg_rem_mask);
        } else {
            mov(reg_shift_table, vlen);
            sub(reg_shift_table, reg_tmp);
            uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
        }
        compute(true);
    }
    L(loop_end);

    postamble();

    for (auto& inj : eltwise_injectors_)
        inj->prepare_table();

    if (utils::one_of(isa, avx2, sse42)) {
        align(64);
        L(l_table);
        for (size_t i = 0; i < vlen; i++) dd(0xFFFFFFFF);
        for (size_t i = 0; i < vlen; i++) dd(0x00000000);
    }

    ker_ = getCode<decltype(ker_)>();
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::operator()(float *dst, const float *bias, const int len, const int oc_start, const int oc_work, const int oc_stride) {
    parallel_nd(oc_work, [&](int oc) {
        ker_args args;
        args.dst = dst + oc * oc_stride;
        args.bias = bias + oc_start + oc;
        args.len = len;
        args.oc_offset = oc_start + oc;
        ker_(&args);
    });
}

void ref_pp_kernel_t::operator()(float *dst, const float *bias, const int len,const int oc_start, const int oc_work, const int oc_stride) {
    // TODO: for "outer threading" we have parallel section within
    // outermost "parallel". It is not good. Consider to use
    // "parallel" here with number of threads passed as parameter
    const auto &p = post_ops_;
    bool need_bias = do_bias_;
    if (use_fast_relu) {
        parallel_nd(oc_work, [&](const int oc) {
            float b = need_bias ? bias[oc_start + oc] : 0;
            float *d_ = dst + oc * oc_stride;
            PRAGMA_OMP_SIMD()
            for (int oS = 0; oS < len; ++oS) {
                d_[oS] += b;
                if (d_[oS] < 0) d_[oS] *= fast_relu_ns;
            }
        });

        need_bias = false;
    } else if (p.len_ > 0) {
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;

        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                parallel_nd(oc_work, [&](const int oc) {
                    float b = need_bias ? bias[oc_start + oc] : 0;
                    float *d_ = dst + oc * oc_stride;
                    PRAGMA_OMP_SIMD()
                    for (int oS = 0; oS < len; ++oS) {
                        d_[oS] += b;
                        d_[oS] = ref_eltwise_impls_[eltwise_inj_idx]->compute_scalar(d_[oS]);
                    }
                });

                eltwise_inj_idx++;
                need_bias = false;
            } else if (post_op.is_depthwise()) {
                auto depthwise_weights = post_op.depthwise.weights_data;
                auto depthwise_bias = post_op.depthwise.biases_data;

                parallel_nd(oc_work, [&](const int oc) {
                    float b = need_bias ? bias[oc_start + oc] : 0;
                    float *d_ = dst + oc * oc_stride;
                    PRAGMA_OMP_SIMD()
                    for (int oS = 0; oS < len; ++oS) {
                        d_[oS] += b;
                        d_[oS] = ref_depthwise_impls_[depthwise_inj_idx]->compute_scalar(d_[oS],
                                                                                        depthwise_weights + oc_start + oc,
                                                                                        depthwise_bias + oc_start + oc);
                    }
                });

                depthwise_inj_idx++;
                need_bias = false;
            } else if (post_op.is_quantization()) {
                auto quant = post_op.quantization;
                auto pcl = quant.crop_low_data->shifts_;
                auto pch = quant.crop_high_data->shifts_;
                auto pisc = quant.input_scale_data->scales_;
                auto pish = quant.input_shift_data->shifts_;
                auto posc = quant.output_scale_data->scales_;
                auto posh = quant.output_shift_data->shifts_;

                parallel_nd(oc_work, [&](const int oc) {
                    float b = need_bias ? bias[oc_start + oc] : 0;
                    float *d_ = dst + oc * oc_stride;

                    int cl_idx = quant.crop_low_data->count_ == 1 ? 0 : oc_start + oc;
                    int ch_idx = quant.crop_high_data->count_ == 1 ? 0 : oc_start + oc;
                    int isc_idx = quant.input_scale_data->count_ == 1 ? 0 : oc_start + oc;
                    int ish_idx = quant.input_shift_data->count_ == 1 ? 0 : oc_start + oc;
                    int osc_idx = quant.output_scale_data->count_ == 1 ? 0 : oc_start + oc;
                    int osh_idx = quant.output_shift_data->count_ == 1 ? 0 : oc_start + oc;

                    PRAGMA_OMP_SIMD()
                    for (int oS = 0; oS < len; ++oS) {
                        d_[oS] += b;

                        d_[oS] = nstl::min(pch[ch_idx], nstl::max(pcl[cl_idx], d_[oS]));
                        d_[oS] = d_[oS] * pisc[isc_idx] + pish[ish_idx];
                        d_[oS] = roundf(d_[oS]);
                        d_[oS] = d_[oS] * posc[osc_idx] + posh[osh_idx];
                    }
                });

                need_bias = false;
            }
        }
    }

    if (need_bias) {
        parallel_nd(oc_work, [&](const int oc) {
            float b = bias[oc_start + oc];
            float *d_ = dst + oc * oc_stride;
            PRAGMA_OMP_SIMD()
            for (int oS = 0; oS < len; ++oS) {
                d_[oS] += b;
            }
        });
    }
};

void gemm_convolution_fwd_t::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t*>(this->memory());

    auto col = scratchpad().get<data_t>(key_conv_gemm_col);

    const auto &jcp = this->pd()->jcp_;
    const int MB = pd()->MB();

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());

    const size_t src_step = (src_d.blk_off(1) - src_d.off_l(0)) / jcp.ngroups;
    const size_t dst_step = (dst_d.blk_off(1) - dst_d.off_l(0)) / jcp.ngroups;
    const size_t weights_oc_size = jcp.ic * jcp.ks;
    const size_t weights_g_size = weights_oc_size * jcp.oc;
    const bool is_problem_3d = pd()->ndims() == 5;
    src += src_d.off_l(0);
    dst += dst_d.off_l(0);

    assert(IMPLICATION(
            is_problem_3d, jcp.os_block == jcp.os && jcp.ic_block == jcp.ic));

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;
        if (is_problem_3d) {
            // jit_gemm_convolution_utils::im2col_3d() requires external
            // data initialization by zeroes
            for (ptrdiff_t i = 0; i < jcp.im2col_sz; i++)
                _col[i] = (data_t)0;
        }

        auto inner_ker = [&](int spatial, const im_pos_t &curr, im_pos_t &prev,
                                 im_pos_t &step, const im_pos_t &end) {
            const data_t *_src
                    = src + (curr.n * jcp.ngroups + curr.g) * src_step;
            step.oc = nstl::min(
                    jcp.oc_block, nstl::min(jcp.oc, end.oc) - curr.oc);
            step.sp = nstl::min(jcp.os_block,
                    nstl::min(jcp.os - curr.sp, end.sp - spatial));
            step.ic = nstl::min(
                    jcp.ic_block, nstl::min(jcp.ic, end.ic) - curr.ic);
            bool do_im2col = curr.do_im2col(prev);
            prev = curr;

            if (jcp.im2col_sz && do_im2col) {
                if (!is_problem_3d)
                    jit_gemm_convolution_utils::im2col<float>(
                            jcp, _src, _col, curr.sp, step.sp, curr.ic, step.ic);
                else
                    jit_gemm_convolution_utils::im2col_3d<float>(
                            jcp, _src, _col, curr.od);
            }
            const data_t one = 1.0;

            const int M = jcp.os * jcp.od;
            const int m = step.sp;
            const int LDA = jcp.im2col_sz ? m : M;

            data_t *_dst = dst + (curr.n * jcp.ngroups + curr.g) * dst_step
                    + curr.oc * M + curr.od * jcp.os + curr.sp;
            const int K = step.ic * jcp.ks;
            const int LDB = jcp.ic * jcp.ks;
            const int N = step.oc;

            // TODO: what if this->beta_ != 0 && != 1 ?
            const float beta = (curr.ic == 0) ? this->beta_ : one;
            const float *_source = jcp.im2col_sz
                    ? _col
                    : _src + curr.ic * M + curr.od * jcp.os + curr.sp;
            const data_t *_weights = weights + curr.g * weights_g_size
                    + curr.oc * weights_oc_size + curr.ic * jcp.ks;

            extended_sgemm("N", "N", &m, &N, &K, &one, _source, &LDA, _weights,
                    &LDB, &beta, _dst, &M);

            if (pp_kernel_ && curr.ic == jcp.ic - step.ic) {
                (*pp_kernel_)(_dst, bias, m, curr.g * jcp.oc + curr.oc, step.oc, M);
            }
        };
        im_pos_t start, end;
        end.ic = jcp.ic;

        if (!is_problem_3d) {
            const int sp_work = MB * jcp.ngroups * jcp.od * jcp.os;
            balance2D(nthr, ithr, sp_work, start.sp, end.sp, jcp.oc, start.oc,
                    end.oc, jcp.nthr_oc);
        } else {
            const int sp_work = MB * jcp.ngroups * jcp.od;
            balance2D(nthr, ithr, sp_work, start.sp, end.sp, jcp.oc, start.oc,
                    end.oc, jcp.nthr_oc);
            start.sp *= jcp.os;
            end.sp *= jcp.os;
        }

        im_pos_t curr, prev, step;
        prev.n = prev.g = prev.od = prev.sp = prev.ic = -1;
        step.oc = jcp.oc_block;
        step.sp = jcp.os_block;
        step.ic = jcp.ic_block;

        if (jcp.loop_order == gemm_loop_rlb)
            for (curr.ic = 0; curr.ic < jcp.ic; curr.ic += step.ic)
                for (int spatial = start.sp; spatial < end.sp;
                        spatial += step.sp) {
                    nd_iterator_init(spatial, curr.n, MB, curr.g,
                            jcp.ngroups, curr.od, jcp.od, curr.sp, jcp.os);
                    for (curr.oc = start.oc; curr.oc < end.oc;
                            curr.oc += step.oc) {
                        inner_ker(spatial, curr, prev, step, end);
                    }
                }
        else if (jcp.loop_order == gemm_loop_lrb)
            for (int spatial = start.sp; spatial < end.sp; spatial += step.sp) {
                nd_iterator_init(spatial, curr.n, MB, curr.g, jcp.ngroups,
                        curr.od, jcp.od, curr.sp, jcp.os);
                for (curr.ic = 0; curr.ic < jcp.ic; curr.ic += step.ic)
                    for (curr.oc = start.oc; curr.oc < end.oc;
                            curr.oc += step.oc)
                        inner_ker(spatial, curr, prev, step, end);
            }
        else
            assert("Unknown loop order");
    });
}

void gemm_convolution_bwd_data_t::execute_backward_data() const {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory());

    auto col = scratchpad().get<data_t>(key_conv_gemm_col);

    const auto &jcp = this->pd()->jcp_;
    const int MB = pd()->MB();

    const int M = jcp.os * jcp.od;
    const size_t src_step_to_clean = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const size_t src_step = (diff_src_d.blk_off(1) - diff_src_d.off_l(0)) / jcp.ngroups;
    const size_t dst_step = (diff_dst_d.blk_off(1) - diff_dst_d.off_l(0)) / jcp.ngroups;
    diff_src += diff_src_d.off_l(0);
    diff_dst += diff_dst_d.off_l(0);

    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int m = jcp.os;
    const int K = jcp.oc;
    const int N = jcp.ic * jcp.ks;
    const int LDC = jcp.im2col_sz ? m : M;

    const size_t work_amount = (size_t)jcp.ngroups * MB;
    const bool is_problem_3d = pd()->ndims() == 5;
    const auto &p = pd()->attr()->post_ops_;

    parallel(jcp.nthr, work_amount, [&](const int ithr, const int nthr) {
        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        int g{0}, n{0};
        size_t start = 0, end = 0;
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, n, MB);
        for (size_t iwork = start; iwork < end; ++iwork) {

            data_t *_diff_src = diff_src + (n * jcp.ngroups + g)*src_step;
            if (is_problem_3d && jcp.im2col_sz > 0) {
                // jit_gemm_convolution_utils::col2im_3d() assumes that the
                // accumulator is initialized by zeroes
                for (size_t i = 0; i < src_step_to_clean; i++)
                    _diff_src[i] = (data_t)0;
            }

            const data_t *_weights = weights + g * weights_g_size;
            for (int od = 0; od < jcp.od; ++od) {
                const data_t *_diff_dst = diff_dst + (n * jcp.ngroups + g)
                    *dst_step + od * m;

                const data_t zero = 0.0, one = 1.0;
                extended_sgemm("N", "T", &m, &N, &K, &one, _diff_dst, &M,
                    _weights, &N, &zero,
                    jcp.im2col_sz ? _col:_diff_src + od * m, &LDC);

                if (jcp.im2col_sz) {
                    if (!is_problem_3d)
                        jit_gemm_convolution_utils::col2im(jcp, _col,
                            _diff_src);
                    else
                        jit_gemm_convolution_utils::col2im_3d(jcp, _col,
                            _diff_src, od);
                }
            }
            if (p.len_ > 0) {
                int depthwise_inj_idx = 0;
                for (int i = 0; i < p.len_; i++) {
                    auto &post_op = p.entry_[i];
                    if (post_op.is_depthwise()) {
                        auto depthwise_weights = post_op.depthwise.weights_data;
                        auto depthwise_bias = post_op.depthwise.biases_data;
                        parallel_nd(jcp.ic, [&](const int ic) {
                            for (int id = 0; id < jcp.id; ++id) {
                                data_t *d_ = _diff_src + ic * jcp.id * jcp.is + id * jcp.is;
                                for (int iS = 0; iS < jcp.is; ++iS) {
                                    d_[iS] = depthwise_injectors[depthwise_inj_idx]->compute_scalar(d_[iS],
                                            depthwise_weights + g * jcp.ic + ic, depthwise_bias + g * jcp.ic + ic);
                                }
                            }
                        });
                        depthwise_inj_idx++;
                    }
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, MB);
        }
    });
}

void gemm_convolution_bwd_weights_t::execute_backward_weights() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    auto col = scratchpad().get<data_t>(key_conv_gemm_col);
    auto wei_reduction = scratchpad().get<data_t>(key_conv_wei_reduction);

    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    const int K = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * K;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int k = jcp.os;
    const int N = jcp.oc;
    const int M = jcp.ic * jcp.ks;
    const int LDA = jcp.im2col_sz ? k : K;
    const bool is_problem_3d = pd()->ndims() == 5;

    parallel(jcp.nthr, jcp.nthr, [&](const int ithr, const int nthr) {
        int ithr_g, nthr_g, ithr_mb, nthr_mb;
        size_t g_start{0}, g_end{0}, mb_start{0}, mb_end{0};

        const int mb_for_balance = jcp.need_wei_reduction ? jcp.mb : 1;
        jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr, jcp.ngroups,
                mb_for_balance, ithr_g, nthr_g, ithr_mb, nthr_mb);

        assert(IMPLICATION(!jcp.need_wei_reduction, nthr_mb == 1));
        const int need_reduction = nthr_mb != 1;

        if (ithr_g != -1 && ithr_mb != -1) {
            balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);
            balance211((size_t)jcp.mb, nthr_mb, ithr_mb, mb_start, mb_end);

            assert(IMPLICATION((g_end - g_start) > 1, need_reduction == 0));

            data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;
            if (is_problem_3d) {
                // jit_gemm_convolution_utils::im2col_3d() requires external
                // data initialization by zeroes
                for (ptrdiff_t i = 0; i < jcp.im2col_sz; i++)
                    _col[i] = (data_t)0;
            }

            data_t *weights_reduce_base = wei_reduction
                    + ithr_g * nthr_mb * weights_g_size;
            data_t *weights_reduce = weights_reduce_base
                    + ithr_mb * weights_g_size;

            for (size_t g = g_start; g < g_end; ++g) {
                data_t *_diff_weights = need_reduction
                        ? weights_reduce : (diff_weights + g * weights_g_size);
                for (size_t mb = mb_start; mb < mb_end; ++mb) {
                    const data_t *_src = src + (mb*jcp.ngroups+g)*src_step;
                    for (int od = 0; od < jcp.od; ++od) {
                    const data_t *_diff_dst = diff_dst
                            + (mb*jcp.ngroups+g)*dst_step + od * k;

                    if (jcp.im2col_sz) {
                        if (!is_problem_3d)
                            jit_gemm_convolution_utils::im2col<float>(
                                    jcp, _src, _col, 0, jcp.os, 0, jcp.ic);
                        else
                            jit_gemm_convolution_utils::im2col_3d<float>(
                                    jcp, _src, _col, od);
                    }

                    const data_t zero = 0.0, one = 1.0;
                    extended_sgemm(
                        "T", "N", &M, &N, &k, &one,
                        jcp.im2col_sz ? _col : _src + od * k,
                        &LDA, _diff_dst, &K,
                        mb == mb_start && od == 0 ? &zero : &one,
                        _diff_weights, &M);
                    }
                }
            }
            if (need_reduction) {
                mkldnn_thr_barrier();
                data_t *weights_base = diff_weights + g_start * weights_g_size;
                jit_gemm_convolution_utils::bwd_weights_reduction_par(
                    ithr_mb, nthr_mb, jcp, weights_reduce_base, weights_base);
            }
        } else
            if (need_reduction) { mkldnn_thr_barrier(); }
    });

    if (jcp.with_bias) {
        parallel_nd(jcp.ngroups, jcp.oc, [&](int g, int oc) {
            data_t db = 0;
            size_t offset_ = (size_t)g * dst_step + (size_t)oc * K;
            for (int mb = 0; mb < jcp.mb; ++mb)
            {
                size_t offset = offset_ + (size_t)mb * jcp.ngroups * dst_step;
                for (int od = 0; od < jcp.od; ++od)
                for (int oh = 0; oh < jcp.oh; ++oh)
                PRAGMA_OMP_SIMD(reduction(+:db))
                for (int ow = 0; ow < jcp.ow; ++ow) {
                    db += diff_dst[offset];
                    offset++;
                }
            }
            diff_bias[g*jcp.oc+oc] = db;
        });
    }
}

template class jit_pp_kernel_t<avx512_common>;
template class jit_pp_kernel_t<avx2>;
template class jit_pp_kernel_t<sse42>;

}
}
}
