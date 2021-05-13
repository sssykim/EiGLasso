# EiGLasso
A scalable optimization method for learning the Kronecker sum of two sparse inverse covariance matrices.

# Installation

## Intel Math Kernel Library

EiGLasso uses Intel Math Kernel Library for linear algebra operations. Please install MKL [here](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html#onemkl) before you complie the EiGLasso software. Then, in Makefile, specify the directory where MKL is installed. Also, it is recommended to use the link advisor from Intel ([link](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html)) to make sure that the software complies correctly. Please refer to [Makefile](Makefile) in this repository as an example.

## Linux

Use Makefile to compile the software.

```bash
make eiglasso
```

# Usage

```bash
eiglasso [input T] [input S] [output psi] [output theta] [output summary] [K] [initial psi] [initial theta] [gamma] [tolerance]
```

- input T, S: Path to the sample covariances for psi and theta. The first line is p or q, which is followed by the actual matrix (white-space separated). [Example](data/T1.txt)
- output psi, theta: Path to save the estimated psi and theta.
- output summary: Path to save the summary of optimization.
- K: Degree of Hessian approximation
- initial psi, theta: Path to the initial psi and theta.
- gamma: Regularization parameter.
- tolerance: Convergence tolerance.


# Example

Demo run of EiGLasso on p=q=100 sample data.

```bash
eiglasso data/T1.txt data/S1.txt psi.txt theta.txt sum.txt 1 data/I100.txt data/I100.txt 0.3 1e-3
```
