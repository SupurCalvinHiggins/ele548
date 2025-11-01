; ModuleID = 'step1.ll'
source_filename = "step0.ll"

define i32 @square(i32 %0) {
  %2 = mul nsw i32 %0, %0
  ret i32 %2
}
