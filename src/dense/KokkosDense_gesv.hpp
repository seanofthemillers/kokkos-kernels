/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

/// \file KokkosDense_gesv.hpp
/// \brief Local dense linear solve
///
/// This file provides KokkosDense::gesv. This function performs a
/// local (no MPI) dense linear solve on a system of linear equations
/// A * X = B where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.

#ifndef KOKKOSDENSE_GESV_HPP_
#define KOKKOSDENSE_GESV_HPP_

#include <type_traits>

#include "KokkosDense_gesv_spec.hpp"

namespace KokkosDense {

/// \brief Solve the dense linear equation system A*X = B.
///
/// \tparam AMatrix Input matrix/Output LU, as a 2-D Kokkos::View.
/// \tparam BXMV Input (right-hand side)/Output (solution) (multi)vector, as a 2-D Kokkos::View.
///
/// \param pivot [in] "Y" (for partial pivoting), or "N" (for no pivoting).
/// \param A [in,out] On entry, the N-by-N matrix to be solved. On exit, the factors L and U from
///   the factorization A = P*L*U; the unit diagonal elements of L are not stored.
/// \param B [in,out] On entry, the right hand side (multi)vector B. On exit, the solution (multi)vector X.
///
template <class AMatrix, class BXMV>
void
trsv (const char pivot[],
      const AMatrix& A,
      const BXMV& B)
{

  static_assert (Kokkos::Impl::is_view<AMatrix>::value,
                 "KokkosDense::gesv: A must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<BXMV>::value,
                 "KokkosDense::gesv: B must be a Kokkos::View.");
  static_assert (static_cast<int> (AMatrix::rank) == 2,
                 "KokkosDense::gesv: A must have rank 2.");
  static_assert (static_cast<int> (BXMV::rank) == 1 || static_cast<int> (BXMV::rank) == 2,
                 "KokkosDense::gesv: B must have either rank 1 or rank 2.");


  // Check validity of pivot argument
  bool valid_pivot = (pivot[0] == 'Y') || (pivot[0] == 'y') ||
                     (pivot[0] == 'N') || (pivot[0] == 'n');
  if(!(valid_pivot)) {
    std::ostringstream os;
    os << "KokkosDense::gesv: pivot[0] = '" << pivot[0] << "'. " <<
      "Valid values include 'N' or 'n' (No pivoting), 'Y' or 'y' (partial pivoting).";
    Kokkos::Impl::throw_runtime_exception (os.str ());
  }

  // Check compatibility of dimensions at run time.
  int64_t A0 = A.extent(0);
  int64_t A1 = A.extent(1);
  int64_t B0 = B.extent(0);
  int64_t B1 = B.extent(1);

  if ( (A0 != A1) ||
       (A1 != B0) ) {
      std::ostringstream os;
      os << "KokkosDense::gesv: Dimensions of A, and B do not match: "
         << " A: " << A.extent(0) << " x " << A.extent(1)
         << " B: " << B.extent(0) << " x " << B.extent(1);
      Kokkos::Impl::throw_runtime_exception (os.str ());
    }

  typedef Kokkos::View<typename AMatrix::const_value_type**,
                       typename AMatrix::array_layout,
                       typename AMatrix::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > AMatrix_Internal;
  typedef Kokkos::View<typename BXMV::const_value_type**,
                       typename BXMV::array_layout,
                       typename BXMV::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > BXMV_Internal;
  AMatrix_Internal A_i = A;
  BXMV_Internal B_i = B;
  
  KokkosDense::Impl::GESV<AMatrix_Internal, BXMV_Internal>::gesv (pivot, A_i, B_i);
}

} // namespace KokkosDense

#endif // KOKKOSDENSE_TRSV_HPP_

