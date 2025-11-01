; ModuleID = 'step0.ll'
source_filename = "step0.ll"

define i32 @square(i32 %0) {
  %2 = add i32 0, 42
  %3 = mul nsw i32 %0, %0
  ret i32 %3
}
