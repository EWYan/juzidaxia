# inline关键字的几种应用场景
---
> 🪶date: 2023/12/31  

## inline关键字的作用
使用关键字inline告诉编译器，需要将目标函数在调用的地方直接展开，而不是进行常规地函数调用，利用空间换时间的策略，提高代码的执行效率。

## 测试inline关键字的使用
*基于TASKING v6.3r1版本编译器实验*🚧

### 实验1🍥
使用inline关键字在源文件中进行函数定义，在其它源文件中使用extern关键字进行声明

```c title="inline_test.c" linenums="1"
#include "Printk.h"
__attribute__((always_inline)) inline
void test_inline (void) {
    unsigned int i = 0;
    for (i = 0; i < 100; i++) {
        printk("test_inline %d\n", i);
    }
}

void foo_main(void) {
    __asm("nop");
    test_inline();
}
```
```c title="inline_test1.c" linenums="1"
__attribute__((always_inline)) inline
extern void test_inline(void);

void bar_main(void) {
    test_inline();
}
```
生成地汇编代码如下，可以看到test_inline函数被直接展开到了foo_main函数中（行3，行5-13），而bar_main函数中则直接调用了test_inline函数（行34）

```asm title="inline_test.s" linenums="1" hl_lines="3 5-13 19-29 34"

Starting disassembly of section 182 (.text.inline_test.foo_main):

	__test_i:	 sub.a	sp,#0x8
	80466ee2:	 nop
	80466ee4:	 mov	d15,#0x0
	80466ee6:	 movh.a	a15,#0x804a
	80466eea:	 lea	a15,[a15]0x4354
	80466eee:	 lea	a12,0x63
	80466ef2:	 st.w	[sp],d15
	80466ef4:	 mov.aa	a4,a15
	80466ef6:	 call	0x8045291c
	80466efa:	 add	d15,#0x1
	80466efc:	 loop	a12,0x80466ef2
	80466efe:	 ret


Starting disassembly of section 181 (.text.inline_test.test_inline):

	__foo_ma:	 sub.a	sp,#0x8
	80466f02:	 mov	d15,#0x0
	80466f04:	 movh.a	a15,#0x804a
	80466f08:	 lea	a15,[a15]0x4354
	80466f0c:	 lea	a12,0x63
	80466f10:	 st.w	[sp],d15
	80466f12:	 mov.aa	a4,a15
	80466f14:	 call	0x8045291c
	80466f18:	 add	d15,#0x1
	80466f1a:	 loop	a12,0x80466f10
	80466f1c:	 ret

Starting disassembly of section 184 (.text.inline_test1.bar_main):

	bar_main:	 nop
	80466f22:	 call	0x80466f00
	80466f26:	 nop
	80466f28:	 ret

```
### 实验2🍥
在头文件中使用inline关键字进行函数定义，在源文件中直接进行调用

```c title="inline_test.h" linenums="1"
#ifndef INLINE_TEST_H
#define INLINE_TEST_H
#include "Printk.h"
__attribute__((always_inline)) inline
void test_inline (void) {
    unsigned int i = 0;
    for (i = 0; i < 100; i++) {
        printk("test_inline %d\n", i);
    }
}

#endif
```
```c title="inline_test.c" linenums="1"
#include "inline_test.h"

void foo_main(void) {
    __asm("nop");
    test_inline();
}
```
```c title="inline_test1.c" linenums="1"
#include "inline_test.h"

void bar_main(void) {
    test_inline();
}
```
使用如上方式进行编译，编译器直接报错，
    
```bash title="编译报错信息"
ltc E108: multiple definitions of symbol "test_inline" in both "inline_test1.o" and "inline_test.o"
```
为啥使用了inline关键字，在代码链接的时候还会存在符号冲突的问题呢？
既然有符号冲突，那么就说明在链接的时候，编译器在链接的时候找到了两个test_inline函数的定义，那么这两个定义分别在哪里呢？查看inline_test.o文件的汇编代码，可以看到在行7-15处foo_main函数对test_inline函数做了内联展开，编译器同时在目标文件中保留了test_inlie函数的函数定义，行22-33处是test_inline函数的定义，同理inline_test1.o文件中也存在test_inline函数的定义，所以在链接的时候就会存在符号冲突的问题。
```bash title="查看inline_test.o文件" linenums="1" hl_lines="7-15 22-33"
#> hldumptc.exe -F2 inline_test.o > inline_test.o.asm
---------- Section dump ----------

                                       .sdecl '.text.inline_test.foo_main', CODE AT 0x0
                                       .sect  '.text.inline_test.foo_main'
00000000 00 00        foo_main:        nop
00000002 da 00                         mov         d15,#0x0
00000004 91 00 00 f0                   movh.a      a15,#0x0
00000008 d9 ff 00 00                   lea         a15,[a15]0x0
0000000c c5 0c 23 10                   lea         a12,0x63
00000010 40 f4                         mov.aa      a4,a15
00000012 02 f4                         mov         d4,d15
00000014 6d 00 00 00                   call        0x14
00000018 c2 1f                         add         d15,#0x1
0000001a fc cb                         loop        a12,0x10
0000001c 00 90        __test_inline_function_end:ret

                .sdecl '.rodata.inline_test..1.str',DATA AT 0x0
                .sect  '.rodata.inline_test..1.str'
                .byte 74,65,73,74,5f,69,6e,6c,69,6e,65,20,25,64,0a,00; test_inline %d..

                                       .sdecl '.text.inline_test.test_inline', CODE AT 0x0
                                       .sect  '.text.inline_test.test_inline'
00000000 da 00        foo_main:        mov         d15,#0x0
00000002 91 00 00 f0                   movh.a      a15,#0x0
00000006 d9 ff 00 00                   lea         a15,[a15]0x0
0000000a c5 0c 23 10                   lea         a12,0x63
0000000e 40 f4                         mov.aa      a4,a15
00000010 02 f4                         mov         d4,d15
00000012 6d 00 00 00                   call        0x12
00000016 c2 1f                         add         d15,#0x1
00000018 fc cb                         loop        a12,0xe
0000001a 00 90                         ret
```
根据GNU手册中的描述：
> When an inline function is not static, then the compiler must assume that there may be calls from other source files;  

不使用static关键字定义的inline函数，编译器会假设在其它源文件中存在对该函数的调用，所以编译器会在目标文件中保留该函数的定义，所以在链接的时候就会存在符号冲突的问题。
### 实验3🍥
在头文件中使用inline关键字进行函数定义，同时使用static关键字进行修饰

```c title="inline_test.h" linenums="1"
#ifndef INLINE_TEST_H
#define INLINE_TEST_H
#include "Printk.h"
__attribute__((always_inline)) inline
static void test_inline (void) {
    unsigned int i = 0;
    for (i = 0; i < 100; i++) {
        printk("test_inline %d\n", i);
    }
}

#endif
```
查看test_inline.o文件的汇编代码，发现test_inline函数的定义已经不存在了，编译器在链接的时候也不会存在符号冲突的问题了。
```bash title="查看inline_test.o文件" linenums="1"
#> hldumptc.exe -F2 inline_test.o > inline_test.o.asm
---------- Section dump ----------

                .sdecl '.rodata.inline_test..1.str',DATA AT 0x0
                .sect  '.rodata.inline_test..1.str'
                .byte 74,65,73,74,5f,69,6e,6c,69,6e,65,20,25,64,0a,00; test_inline %d..

                                       .sdecl '.text.inline_test.foo_main', CODE AT 0x0
                                       .sect  '.text.inline_test.foo_main'
00000000 00 00        foo_main:        nop
00000002 da 00                         mov         d15,#0x0
00000004 91 00 00 f0                   movh.a      a15,#0x0
00000008 d9 ff 00 00                   lea         a15,[a15]0x0
0000000c c5 0c 23 10                   lea         a12,0x63
00000010 40 f4                         mov.aa      a4,a15
00000012 02 f4                         mov         d4,d15
00000014 6d 00 00 00                   call        0x14
00000018 c2 1f                         add         d15,#0x1
0000001a fc cb                         loop        a12,0x10
0000001c 00 90                         ret
```

### 总结📋
- inline关键字的作用是告诉编译器，需要将目标函数在调用的地方直接展开，而不是进行常规地函数调用，利用空间换时间的策略，提高代码的执行效率。
- inline关键字的使用场景：
    - 在头文件中使用inline关键字进行函数定义，同时使用static关键字进行修饰
    - 在源文件中使用inline关键字进行函数定义，在其它源文件中使用extern关键字进行声明，定义处源文件中使用目标inline函数会被直接展开，而其它源文件中使用目标inline函数会被直接调用

## 参考资料
- [TASKING编译器手册](https://www.tasking.com/support/tricore/ctc_user_guide_v6.2r1.pdf)
- [GNU在线文档](https://gcc.gnu.org/onlinedocs/gcc-6.1.0/gcc/Inline.html)

---