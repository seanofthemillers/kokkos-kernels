/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#ifndef _KOKKOSSPARSE_UTILS_ROCSPARSE_HPP
#define _KOKKOSSPARSE_UTILS_ROCSPARSE_HPP

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#ifndef HIP_VERSION_MAJOR
#error "Missing HIP Version definition!"
#endif
#if (HIP_VERSION_MAJOR < 5) || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR < 2)
#include <rocsparse.h>
#else
#include <rocsparse/rocsparse.h>
#endif

namespace KokkosSparse {
namespace Impl {

inline void rocsparse_internal_error_throw(rocsparse_status rocsparseStatus,
                                           const char* name, const char* file,
                                           const int line) {
  std::ostringstream out;
  out << name << " error( ";
  switch (rocsparseStatus) {
    case rocsparse_status_invalid_handle:
      out << "rocsparse_status_invalid_handle): handle not initialized, "
             "invalid or null.";
      break;
    case rocsparse_status_not_implemented:
      out << "rocsparse_status_not_implemented): function is not implemented.";
      break;
    case rocsparse_status_invalid_pointer:
      out << "rocsparse_status_invalid_pointer): invalid pointer parameter.";
      break;
    case rocsparse_status_invalid_size:
      out << "rocsparse_status_invalid_size): invalid size parameter.";
      break;
    case rocsparse_status_memory_error:
      out << "rocsparse_status_memory_error): failed memory allocation, copy, "
             "dealloc.";
      break;
    case rocsparse_status_internal_error:
      out << "rocsparse_status_internal_error): other internal library "
             "failure.";
      break;
    case rocsparse_status_invalid_value:
      out << "rocsparse_status_invalid_value): invalid value parameter.";
      break;
    case rocsparse_status_arch_mismatch:
      out << "rocsparse_status_arch_mismatch): device arch is not supported.";
      break;
    case rocsparse_status_zero_pivot:
      out << "rocsparse_status_zero_pivot): encountered zero pivot.";
      break;
    case rocsparse_status_not_initialized:
      out << "rocsparse_status_not_initialized): descriptor has not been "
             "initialized.";
      break;
    case rocsparse_status_type_mismatch:
      out << "rocsparse_status_type_mismatch): index types do not match.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

inline void rocsparse_internal_safe_call(rocsparse_status rocsparseStatus,
                                         const char* name,
                                         const char* file = nullptr,
                                         const int line   = 0) {
  if (rocsparse_status_success != rocsparseStatus) {
    rocsparse_internal_error_throw(rocsparseStatus, name, file, line);
  }
}

// The macro below defines is the public interface for the safe cusparse calls.
// The functions themselves are protected by impl namespace.
#define KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(call)                             \
  KokkosSparse::Impl::rocsparse_internal_safe_call(call, #call, __FILE__, \
                                                   __LINE__)

inline rocsparse_operation mode_kk_to_rocsparse(const char kk_mode[]) {
  rocsparse_operation myRocsparseOperation;
  switch (toupper(kk_mode[0])) {
    case 'N': myRocsparseOperation = rocsparse_operation_none; break;
    case 'T': myRocsparseOperation = rocsparse_operation_transpose; break;
    case 'H':
      myRocsparseOperation = rocsparse_operation_conjugate_transpose;
      break;
    default: {
      std::cerr << "Mode " << kk_mode[0] << " invalid for rocSPARSE SpMV.\n";
      throw std::invalid_argument("Invalid mode");
    }
  }
  return myRocsparseOperation;
}

template <typename index_type>
inline rocsparse_indextype rocsparse_index_type() {
  if (std::is_same<index_type, uint16_t>::value) {
    return rocsparse_indextype_u16;
  } else if (std::is_same<index_type, int32_t>::value) {
    return rocsparse_indextype_i32;
  } else if (std::is_same<index_type, int64_t>::value) {
    return rocsparse_indextype_i64;
  } else {
    std::ostringstream out;
    out << "Trying to call rocSPARSE SpMV with unsupported index type: "
        << typeid(index_type).name();
    throw std::logic_error(out.str());
  }
}

template <typename data_type>
inline rocsparse_datatype rocsparse_compute_type() {
  std::ostringstream out;
  out << "Trying to call rocSPARSE SpMV with unsupported compute type: "
      << typeid(data_type).name();
  throw std::logic_error(out.str());
}

template <>
inline rocsparse_datatype rocsparse_compute_type<float>() {
  return rocsparse_datatype_f32_r;
}

template <>
inline rocsparse_datatype rocsparse_compute_type<double>() {
  return rocsparse_datatype_f64_r;
}

template <>
inline rocsparse_datatype rocsparse_compute_type<Kokkos::complex<float>>() {
  return rocsparse_datatype_f32_c;
}

template <>
inline rocsparse_datatype rocsparse_compute_type<Kokkos::complex<double>>() {
  return rocsparse_datatype_f64_c;
}

}  // namespace Impl

template<typename scalar_type, typename ordinal_type, typename size_type>
inline bool rocsparse_check_valid_spmv_types(bool throw_if_invalid=false) {

  using nonconst_scalar_type  = typename std::remove_const<scalar_type>::type;
  using nonconst_ordinal_type = typename std::remove_const<ordinal_type>::type;
  using nonconst_size_type    = typename std::remove_const<size_type>::type;

  bool valid = true;
  // Make sure input ordinal type is valid for rocsparse (no unsigned ints allowed)
  if(!(std::is_same<nonconst_size_type, int32_t>::value ||
       std::is_same<nonconst_size_type, int64_t>::value)){
    valid = false;
    if(throw_if_invalid)
      throw std::runtime_error("rocsparse_check_valid_spmv_types : Size type (row ptrs) must be int32 or int64");
  }

  // Make sure input ordinal type is valid for rocsparse (no unsigned ints allowed)
  if(!(std::is_same<nonconst_ordinal_type, int32_t>::value ||
       std::is_same<nonconst_ordinal_type, int64_t>::value)){
    valid = false;
    if(throw_if_invalid)
      throw std::runtime_error("rocsparse_check_valid_spmv_types : Ordinal type (column indexes) must be of type int32 or int64");
  }

  // Only certain compute types are supported by rocsparse
  if(!(std::is_same<nonconst_scalar_type, float>::value ||
       std::is_same<nonconst_scalar_type, double>::value)){
    valid = false;
    if(throw_if_invalid)
      throw std::runtime_error("rocsparse_check_valid_spmv_types : Scalar type (values) must be float or double");
  }

  return valid;
}

struct
rocsparseSpMatrixDescriptor
{

  //! Constructor - rocsparse descriptors are not initialized (must call setup)
  rocsparseSpMatrixDescriptor():
    descriptor(0),
    transpose_descriptor(0)
  {

  }

  //! Destructor - destroys rocsparse descriptors
  ~rocsparseSpMatrixDescriptor()
  {
    clear();
  }

  //! Safely destroy the descriptors
  void
  clear()
  {
    if(descriptor){
      rocsparse_destroy_spmat_descr(descriptor);
    }
    if(transpose_descriptor){
      rocsparse_destroy_spmat_descr(transpose_descriptor);
    }
  }

  /**
   * @brief Main setup call for creating a rocsparse sparse matrix descriptor
   * 
   * @note If types are incorrect for rocSPARSE, the descriptors will be set to 0
   * @throws If array sizes are mismatched
   * 
   * @tparam RowPtrArray Implicit template for input CRS row ptrs array [size = num_rows+1]
   * @tparam ColumnIndexArray Implicit template for input CRS column indexes array [size = number of non-zeros]
   * @tparam ValuesArray Implicit template for input CRS values array [size = number of non-zeros]
   * @param num_rows Number of total rows in the matrix
   * @param num_columns Number of total columns in the matrix
   * @param row_ptrs Row pointers for CRS format (i.e. offsets into column_indexes/values arrays for each row)
   * @param column_indexes Column indexes associated with each value in a row in CRS format
   * @param values Values contained in the sparse matrix in CRS format
   */
  template<typename RowPtrArray, typename ColumnIndexArray, typename ValuesArray>
  void
  setup(const rocsparse_int num_rows,
        const rocsparse_int num_columns,
        RowPtrArray row_ptrs,
        ColumnIndexArray column_indexes,
        ValuesArray values)
  {
    
    // Typedefs
    using row_ptrs_type = typename RowPtrArray::non_const_value_type;
    using column_indexes_type = typename ColumnIndexArray::non_const_value_type;
    using compute_type = typename ValuesArray::non_const_value_type;

    // If setup has already been called, reset the existing descriptors
    if(descriptor != 0 || transpose_descriptor != 0)
      clear();

    // Check that memory space is HIP for all input arrays
    if(!(std::is_same<typename RowPtrArray::memory_space,      Kokkos::Experimental::HIPSpace>::value &&
         std::is_same<typename ColumnIndexArray::memory_space, Kokkos::Experimental::HIPSpace>::value &&
         std::is_same<typename ValuesArray::memory_space,      Kokkos::Experimental::HIPSpace>::value))
      return;

    // Check datatypes
    if(!rocsparse_check_valid_spmv_types<compute_type, column_indexes_type, row_ptrs_type>())
      return;
    
    // Make sure number of rows is consistent
    if(row_ptrs.size() != size_t(num_rows+1)){
      std::ostringstream out;
      out << "rocsparseSpMatrixDescriptor : Number of rows ("<<num_rows<<") is not consistent with row_ptrs array (size = "<<row_ptrs.size()<<", expected "<<num_rows+1<<")";
      throw std::runtime_error(out.str());
    }

    // Make sure number of nonzeros is consistent
    if(column_indexes.size() != values.size()){
      std::ostringstream out;
      out << "rocsparseSpMatrixDescriptor : Number of non-zeros mismatch between column_indexes (size = "<<column_indexes.size()<<") and values (size = "<<values.size()<<")";
      throw std::runtime_error(out.str());
    }

    auto rocsparse_row_ptrs_type = KokkosSparse::Impl::rocsparse_index_type<row_ptrs_type>();
    auto rocsparse_column_indexes_type = KokkosSparse::Impl::rocsparse_index_type<column_indexes_type>();
    auto rocsparse_compute_type = KokkosSparse::Impl::rocsparse_compute_type<compute_type>();

    const rocsparse_int nnz = values.size();

    // Even though these values will not be modified by rocsparse, they must be passed in as non-const void*
    void* csr_row_ptr =
        static_cast<void*>(const_cast<row_ptrs_type*>(row_ptrs.data()));
    void* csr_col_ind =
        static_cast<void*>(const_cast<column_indexes_type*>(column_indexes.data()));
    void* csr_val =
        static_cast<void*>(const_cast<compute_type*>(values.data()));

    // Regular descriptor
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_create_csr_descr(
        &descriptor, num_rows, num_columns, nnz,
        csr_row_ptr, csr_col_ind, csr_val,
        rocsparse_row_ptrs_type, rocsparse_column_indexes_type, rocsparse_index_base_zero, rocsparse_compute_type));

    // Note: Currently rocsparse cannot share descriptors for transpose SpMV using the adaptive scheme
    // Transpose descriptor
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_create_csr_descr(
        &transpose_descriptor, num_rows, num_columns, nnz,
        csr_row_ptr, csr_col_ind, csr_val,
        rocsparse_row_ptrs_type, rocsparse_column_indexes_type, rocsparse_index_base_zero, rocsparse_compute_type));

  }

  rocsparse_spmat_descr descriptor;
  rocsparse_spmat_descr transpose_descriptor;
};

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#endif  // _KOKKOSSPARSE_UTILS_ROCSPARSE_HPP
