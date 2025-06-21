add_mlir_library(MLIRLinalgGenericOptimization
  LinalgGenericOptimization.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math

  DEPENDS
  MLIRPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRMathDialect
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
)
