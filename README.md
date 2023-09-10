# Generates wrappers for Math Kernel Library routines

## Installation

The package relies on [modernc.org/cc/v4](https://pkg.go.dev/modernc.org/cc/v4) as AST parser, which is a pure go implementation. The binary can be installed with `go install`:

```shell
go install github.com/fardream/gen-mkl-wrapper@latest
```

## Wrappers

Math Kernel Library by Intel is widely used library of common mathematical routines, which provides support for various BLAS and LAPACK routines and many many more.
Due to its root in Fortran and C, many routines' names contain the type its operates on, for example, the Cholesky Decomposition is

```C
lapack_int LAPACKE_spotrf (int matrix_layout , char uplo , lapack_int n , float * a , lapack_int lda ); // for float, or 32-bit float point number.
lapack_int LAPACKE_dpotrf (int matrix_layout , char uplo , lapack_int n , double * a , lapack_int lda ); // for double, or 64-bit float point number.
```

For C++ or rust, it may be desired to dispatch the method based on the type. This becomes extremely handy when implementing something based on MKL for both 32-bit and 64-bit float point number.

In C++, below is valid

```C++
lapack_int LAPACKE_potrf (int matrix_layout , char uplo , lapack_int n , float * a , lapack_int lda ); // for float, or 32-bit float point number.
lapack_int LAPACKE_potrf (int matrix_layout , char uplo , lapack_int n , double * a , lapack_int lda ); // for double, or 64-bit float point number.
```

Similarly in rust, the below is valid

```rust
pub trait MKLRoutines {
    fn LAPACKE_potrf(matrix_layout: i32, uplo: i8, n: i32, a: *mut Self, lda: i32) -> i32;
}

impl MKLRoutines for f64 {
    // for f64, or 64-bit float point number.
    fn LAPACKE_potrf(matrix_layout: i32, uplo: i8, n: i32, a: *mut Self, lda: i32) -> i32 {
        unsafe {
            LAPACKE_dpotrf(matrix_layout, uplo, n, a, lda)
        }
    }
}

impl MKLRoutines for f32 {
    // for f32, or 32-bit float point number.
    fn LAPACKE_potrf(matrix_layout: i32, uplo: i8, n: i32, a: *mut Self, lda: i32) -> i32 {
        unsafe {
            LAPACKE_spotrf(matrix_layout, uplo, n, a, lda)
        }
    }
}
```

This simply binary does just the above - provided with a list of routines names, it will generate the rust trait or C++ polymorphic functions to those routines.

Those generated functions can be called from template/generic functions

```C++
template <typename T>
void SomeFunctionForFloatPoint(T* x) {
    // ... other codes
    LAPACKE_potrf(LAPACK_ROW_MAJOR, 'U', n, x, lda);
    // ... other codes
}
```

or

```rust
pub fn some_function_for_float<T: MKLRoutines>(x:&mut [T]) {
    // ... other codes
    unsafe { T::LAPACKE_potrf(LAPACK_ROW_MAJOR as i32, 'U' as i8, n x.as_mut_ptr(), lda) };
    // ... other codes
}
```
