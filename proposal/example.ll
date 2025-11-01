define i32 @square(i32 %0) {
    %2 = alloca i32, align 4
    store i32 %0, ptr %2, align 4
    %3 = load i32, ptr %2, align 4
    %4 = load i32, ptr %2, align 4
    %5 = mul nsw i32 %3, %4
    ret i32 %5
}

