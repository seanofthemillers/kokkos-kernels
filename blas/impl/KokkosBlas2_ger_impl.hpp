//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOSBLAS2_GER_IMPL_HPP_
#define KOKKOSBLAS2_GER_IMPL_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"
#include "Kokkos_ArithTraits.hpp"

namespace KokkosBlas {
namespace Impl {

// Functor for a single-level parallel_for version of nontranspose GER.
// The functor parallelizes over rows of the input matrix A.
template <class XViewType, class YViewType, class AViewType, class IndexType>
struct SingleLevelGER {
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using XComponentType = typename XViewType::non_const_value_type;
  using YComponentType = typename YViewType::non_const_value_type;
  using AComponentType = typename AViewType::non_const_value_type;

  SingleLevelGER( const bool             justTranspose
                , const AlphaCoeffType & alpha
                , const XViewType      & x
                , const YViewType      & y
                , const AViewType      & A
                )
      : justTranspose_(justTranspose)
      , alpha_        (alpha)
      , x_            (x)
      , y_            (y)
      , A_            (A)
  {
    // Nothing to do
  }

  KOKKOS_INLINE_FUNCTION void operator()(const IndexType & i) const {
    if (alpha_ == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
      // Nothing to do
    }
    else {
      const IndexType      N      ( A_.extent(1) );
      const XComponentType x_fixed( x_(i) );

      if (justTranspose_) {
        for (IndexType j = 0; j < N; ++j) {
          A_(i,j) += AComponentType( alpha_ * x_fixed * y_(j) );
        }
      }
      else {
        for (IndexType j = 0; j < N; ++j) {
          A_(i,j) += AComponentType( alpha_ * x_fixed * Kokkos::ArithTraits<YComponentType>::conj( y_(j) ) );
        }
      }
    }
  }

private:
  bool                           justTranspose_;
  AlphaCoeffType                 alpha_;
  typename XViewType::const_type x_;
  typename YViewType::const_type y_;
  AViewType                      A_;
};

// Single-level parallel version of GER.
template <class ExecutionSpace, class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void singleLevelGer( const          ExecutionSpace              & space
                   , const          char                          trans[]
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  static_assert(std::is_integral<IndexType>::value, "IndexType must be an integer");

  using AlphaCoeffType = typename AViewType::non_const_value_type;

  if (y.extent(0) == 0) {
    // no entries to update
  }
  else if (x.extent(0) == 0) {
    // no entries to update
  }
  else if (alpha == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
    // no entries to update
  }
  else {
    Kokkos::RangePolicy<ExecutionSpace, IndexType> rangePolicy(space, 0, A.extent(0));
    SingleLevelGER<XViewType, YViewType, AViewType, IndexType> functor( (trans[0] == 'T') || (trans[0] == 't')
                                                                      , alpha
                                                                      , x
                                                                      , y
                                                                      , A
                                                                      );
    Kokkos::parallel_for("KokkosBlas::ger[SingleLevel]", rangePolicy, functor);
  }
}

struct TwoLevelGER_LayoutLeftTag {};
struct TwoLevelGER_LayoutRightTag {};

// ---------------------------------------------------------------------------------------------

// Functor for a two-level parallel_reduce version of GER, designed for performance on GPU.
// Kernel depends on the layout of A.
template <class ExecutionSpace, class XViewType, class YViewType, class AViewType, class IndexType>
struct TwoLevelGER {
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using XComponentType = typename XViewType::non_const_value_type;
  using YComponentType = typename YViewType::non_const_value_type;
  using AComponentType   = typename AViewType::non_const_value_type;

  using policy_type    = Kokkos::TeamPolicy<ExecutionSpace>;
  using member_type    = typename policy_type::member_type;

  TwoLevelGER( const bool             justTranspose
             , const AlphaCoeffType & alpha
             , const XViewType      & x
             , const YViewType      & y
             , const AViewType      & A
             )
      : justTranspose_(justTranspose)
      , alpha_        (alpha)
      , x_            (x)
      , y_            (y)
      , A_            (A)
  {
    // Nothing to do
  }

public:
  // LayoutLeft version: one team per column
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutLeftTag
                                        , const member_type & team
                                        ) const {
    if (alpha_ == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
      // Nothing to do
    }
    else {
      const IndexType M ( A_.extent(0) );
      const IndexType j ( team.league_rank() );
      if (justTranspose_) {
        const YComponentType y_fixed( y_(j) );
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M), [&](const IndexType & i) {
          A_(i,j) += AComponentType( alpha_ * x_(i) * y_fixed );
        });
      }
      else {
        const YComponentType y_fixed( Kokkos::ArithTraits<YComponentType>::conj( y_(j) ) );
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M), [&](const IndexType & i) {
          A_(i,j) += AComponentType( alpha_ * x_(i) * y_fixed );
        });
      }
    }
  }

  // LayoutRight version: one team per row
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutRightTag
                                        , const member_type & team
                                        ) const {
    if (alpha_ == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
      // Nothing to do
    }
    else {
      const IndexType      N      ( A_.extent(1) );
      const IndexType      i      ( team.league_rank() );
      const XComponentType x_fixed( x_(i) );
      if (justTranspose_) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const IndexType & j) {
          A_(i,j) += AComponentType( alpha_ * x_fixed * y_(j) );
        });
      }
      else {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const IndexType & j) {
          A_(i,j) += AComponentType( alpha_ * x_fixed * Kokkos::ArithTraits<AlphaCoeffType>::conj( y_(j) ) );
        });
      }
    }
    team.team_barrier();
  }

private:
  bool                           justTranspose_;
  AlphaCoeffType                 alpha_;
  typename XViewType::const_type x_;
  typename YViewType::const_type y_;
  AViewType                      A_;
};

// Two-level parallel version of GER.
template <class ExecutionSpace, class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void twoLevelGer( const          ExecutionSpace              & space
                , const          char                          trans[]
                , const typename AViewType::const_value_type & alpha
                , const          XViewType                   & x
                , const          YViewType                   & y
                , const          AViewType                   & A
                ) {
  static_assert(std::is_integral<IndexType>::value, "IndexType must be an integer");

  using AlphaCoeffType = typename AViewType::non_const_value_type;

  if (y.extent(0) == 0) {
    // no entries to update
    return;
  }
  else if (x.extent(0) == 0) {
    // no entries to update
    return;
  }
  else if (alpha == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
    // no entries to update
    return;
  }

  constexpr bool isLayoutLeft = std::is_same<typename AViewType::array_layout, Kokkos::LayoutLeft>::value;
  using layout_tag  = typename std::conditional<isLayoutLeft, TwoLevelGER_LayoutLeftTag, TwoLevelGER_LayoutRightTag>::type;
  using TeamPolicyType = Kokkos::TeamPolicy<ExecutionSpace, layout_tag>;
  TeamPolicyType teamPolicy;
  if (isLayoutLeft) {
    // LayoutLeft: one team per column
    teamPolicy = TeamPolicyType(space, A.extent(1), Kokkos::AUTO);
  }
  else {
    // LayoutRight: one team per row
    teamPolicy = TeamPolicyType(space, A.extent(0), Kokkos::AUTO);
  }

  TwoLevelGER<ExecutionSpace, XViewType, YViewType, AViewType, IndexType> functor( (trans[0] == 'T') || (trans[0] == 't')
                                                                                 , alpha
                                                                                 , x
                                                                                 , y
                                                                                 , A
                                                                                 );
  Kokkos::parallel_for("KokkosBlas::ger[twoLevel]", teamPolicy, functor);
}

// ---------------------------------------------------------------------------------------------

// generalGer: use 1 level (Range) or 2 level (Team) implementation,
// depending on whether execution space is CPU or GPU.
// The 'enable_if' makes sure unused kernels are not instantiated.

template < class ExecutionSpace
         , class XViewType
         , class YViewType
         , class AViewType
         , class IndexType
         , typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<ExecutionSpace>()>::type* = nullptr
         >
void generalGerImpl( const          ExecutionSpace              & space
                   , const          char                          trans[]
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  singleLevelGer(space, trans, alpha, x, y, A);
}

template < class ExecutionSpace
         , class XViewType
         , class YViewType
         , class AViewType
         , class IndexType
         , typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<ExecutionSpace>()>::type* = nullptr
         >
void generalGerImpl( const          ExecutionSpace              & space
                   , const          char                          trans[]
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  twoLevelGer(space, trans, alpha, x, y, A);
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS2_GER_IMPL_HPP_
