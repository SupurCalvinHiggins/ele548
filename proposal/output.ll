; ModuleID = 'example.ll'
source_filename = "example.ll"

define i32 @square(i32 %0) {
  %2 = mul nsw i32 %0, %0
  ret i32 %2
}
