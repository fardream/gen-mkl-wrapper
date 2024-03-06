// gen-mkl-wrapper generates wrappers for Math Kernel Library routines.
//
// Math Kernel Library by Intel is widely used library of common mathematical routines, which provides support for various BLAS and LAPACK routines and many many more.
// Due to its root in Fortran and C, many routines' names contain the type its operates on, for example, the Cholesky Decomposition is
//
//	lapack_int LAPACKE_spotrf (int matrix_layout , char uplo , lapack_int n , float * a , lapack_int lda ); // for float, or 32-bit float point number.
//	lapack_int LAPACKE_dpotrf (int matrix_layout , char uplo , lapack_int n , double * a , lapack_int lda ); // for double, or 64-bit float point number.
//
// For C++ or rust, it may be desired to dispatch the method based on the type. This becomes extremely handy when implementing something based on MKL for both 32-bit and 64-bit float point number.
//
// For in C++, below is valid
//
//	lapack_int LAPACKE_potrf (int matrix_layout , char uplo , lapack_int n , float * a , lapack_int lda ); // for float, or 32-bit float point number.
//	lapack_int LAPACKE_potrf (int matrix_layout , char uplo , lapack_int n , double * a , lapack_int lda ); // for double, or 64-bit float point number.
//
// Similarly in rust, the below is valid
//
//	pub trait MKLRoutines {
//	    fn LAPACKE_potrf(matrix_layout: i32, uplo: i8, n: i32, a: *mut Self, lda: i32) -> i32;
//	}
//
//	impl MKLRoutines for f64 {
//	    // for f64, or 64-bit float point number.
//	    fn LAPACKE_potrf(matrix_layout: i32, uplo: i8, n: i32, a: *mut Self, lda: i32) -> i32 {
//	        unsafe {
//	            LAPACKE_dpotrf(matrix_layout, uplo, n, a, lda)
//	        }
//	    }
//	}
//
//	impl MKLRoutines for f32 {
//	    // for f32, or 32-bit float point number.
//	    fn LAPACKE_potrf(matrix_layout: i32, uplo: i8, n: i32, a: *mut Self, lda: i32) -> i32 {
//	        unsafe {
//	            LAPACKE_spotrf(matrix_layout, uplo, n, a, lda)
//	        }
//	    }
//	}
//
// This simply binary does just the above - provided with a list of routines names, it will generate the rust trait or C++ polymorphic functions to those routines.
package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"
	"path"
	"sort"
	"strings"
	"text/template"

	"github.com/spf13/cobra"
	"modernc.org/cc/v4"
	"mvdan.cc/gofumpt/format"
)

//go:embed rs.tmpl
var rsTmplText string

//go:embed cc.tmpl
var ccTmplText string

//go:embed go.tmpl
var goTmplText string

var (
	mklPath          = ""
	inputFuncsPath   = ""
	outputFile       = ""
	mklProviderCrate = "crate"
	traitName        = "MKLRoutines"
	forC             = false
	forGo            = false
	goPackageName    = "mklroutines"
)

type funcArg struct {
	name     string
	typeName string
	// rustName is the rust type name
	rustName string
	// dontUse indicates if the type should be imported from crate, for rust
	dontUse bool
	// cgoType is the cgoType that is the input to the code
	cgoType string
	// goType
	goType string
}

type funcDef struct {
	RawName    string
	is32       bool
	ReturnType string
	args       []funcArg
	BetterName string
}

func (f *funcDef) GoName() string {
	name := []byte(f.BetterName)
	if name[0] >= 'a' && name[0] <= 'z' {
		name[0] = 'A' + (name[0] - 'a')
	}
	return string(name)
}

type tmplInput struct {
	funcDefs        []funcDef
	providerCrate   string
	DesiredFuncList []string
}

func (*tmplInput) TraitName() string {
	return traitName
}

func (*tmplInput) GoPackageName() string {
	return goPackageName
}

func (f *funcDef) CParams() string {
	ps := []string{}

	for _, p := range f.args {
		if strings.HasSuffix(p.typeName, "[]") {
			ps = append(ps, fmt.Sprintf("%s %s[]", strings.TrimSuffix(p.typeName, "[]"), p.name))
		} else {
			ps = append(ps, fmt.Sprintf("%s %s", p.typeName, p.name))
		}
	}

	return strings.Join(ps, ",")
}

func (f *funcDef) CInput() string {
	ps := []string{}
	for _, p := range f.args {
		ps = append(ps, p.name)
	}

	return strings.Join(ps, ",")
}

func (i *tmplInput) UseLine() string {
	uses := make([]string, 0, len(i.funcDefs)+3)

	blastypes := make(map[string]struct{})

	for _, f := range i.funcDefs {
		uses = append(uses, f.RawName)
		for _, arg := range f.args {
			if !arg.dontUse {
				blastypes[arg.rustName] = struct{}{}
			}
		}
	}

	for k := range blastypes {
		uses = append(uses, k)
	}

	sort.Strings(uses)

	return fmt.Sprintf("%s::{%s}", i.providerCrate, strings.Join(uses, ", "))
}

type GoFuncPair struct {
	Float64Func *funcDef
	Float32Func *funcDef
	Name        string
}

func (i *tmplInput) GoFuncs() []*GoFuncPair {
	f32funcs := i.F32Funcs()
	result := make([]*GoFuncPair, 0, len(f32funcs))
	byname := make(map[string]*GoFuncPair)

	for _, f32func := range f32funcs {
		f := &GoFuncPair{
			Float32Func: f32func,
			Name:        f32func.GoName(),
		}
		result = append(result, f)
		byname[f.Name] = f
	}

	for _, f64func := range i.F64Funcs() {
		f, ok := byname[f64func.GoName()]
		if !ok {
			panic(fmt.Sprintf("f64 has name %s", f64func.GoName()))
		}
		f.Float64Func = f64func
	}

	return result
}

func getGoParamType(t string) string {
	switch t {
	case "size_t":
		return "uint64"
	case "int32_t", "int", "const int", "const int32_t":
		return "int32"
	case "int64_t":
		return "int64"
	case "const double *", "const float *", "const float[]", "const double[]":
		return "*F"
	case "double *", "float *", "float[]", "double[]":
		return "*F"
	case "double", "float", "const double", "const float":
		return "F"
	case "char":
		return "byte"
	case "int *":
		return "*int32"
	case "const int *":
		return "*int32"
	case "CBLAS_LAYOUT", "CBLAS_UPLO", "CBLAS_DIAG", "CBLAS_TRANSPOSE", "CBLAS_SIDE":
		return t
	}

	if strings.HasPrefix(t, "const ") {
		t = strings.TrimPrefix(t, "const ")
	}

	return fmt.Sprintf("C.%s", t)
}

func (f *GoFuncPair) Params() []string {
	r := []string{}
	for _, p := range f.Float32Func.args {
		t := getGoParamType(p.typeName)
		r = append(r, fmt.Sprintf("%s %s", p.name, t))
	}

	return r
}

func (i *tmplInput) getfuncs(is32 bool) []*funcDef {
	r := []*funcDef{}
	for _, f := range i.funcDefs {
		f := f
		if f.is32 == is32 {
			r = append(r, &f)
		}
	}

	return r
}

func (i *tmplInput) F64Funcs() []*funcDef {
	return i.getfuncs(false)
}

func (i *tmplInput) F32Funcs() []*funcDef {
	return i.getfuncs(true)
}

func (i *tmplInput) TraitFuncs() []*funcDef {
	return i.getfuncs(true)
}

func (f *GoFuncPair) GoReturn() string {
	switch f.Float32Func.ReturnType {
	case "void":
		return ""
	case "int32_t", "int":
		return "int32"
	case "float", "double":
		return "F"
	case "size_t":
		return "uint64"
	default:
		return f.Float32Func.ReturnType
	}
}

func (f *funcDef) ReturnDeclare() string {
	switch f.ReturnType {
	case "void":
		return ""
	case "int32_t", "int":
		return "-> i32"
	case "float", "double":
		return "-> Self"
	case "size_t":
		return "-> usize"
	default:
		return f.ReturnType
	}
}

func getRustParamType(t string) (string, bool) {
	switch t {
	case "size_t":
		return "usize", true
	case "int32_t", "int", "const int", "const int32_t":
		return "i32", true
	case "int64_t":
		return "i64", true
	case "const double *", "const float *", "const float[]", "const double[]":
		return "*const Self", true
	case "double *", "float *", "float[]", "double[]":
		return "*mut Self", true
	case "double", "float", "const double", "const float":
		return "Self", true
	case "char":
		return "i8", true
	case "int *":
		return "*mut i32", true
	case "const int *":
		return "*const i32", true
	}

	if strings.HasPrefix(t, "const ") {
		return strings.TrimPrefix(t, "const "), false
	}

	return t, false
}

func (f *funcDef) Params() []string {
	r := []string{}

	for _, p := range f.args {
		t, _ := getRustParamType(p.typeName)
		r = append(r, fmt.Sprintf("%s: %s", p.name, t))
	}

	return r
}

func (f *funcDef) CallParams() []string {
	r := []string{}

	for _, p := range f.args {
		r = append(r, p.name)
	}

	return r
}

func retrieveType(r *cc.DeclarationSpecifiers) string {
	switch r.Case {
	case cc.DeclarationSpecifiersTypeQual:
		return r.TypeQualifier.Token.SrcStr() + " " + retrieveType(r.DeclarationSpecifiers)
	case cc.DeclarationSpecifiersTypeSpec:
		return r.TypeSpecifier.Token.SrcStr()
	case cc.DeclarationSpecifiersAlignSpec:
		fallthrough
	case cc.DeclarationSpecifiersFunc:
		fallthrough
	case cc.DeclarationSpecifiersStorage:
		fallthrough
	case cc.DeclarationSpecifiersAttr:
		fallthrough
	default:
		return retrieveType(r.DeclarationSpecifiers)
	}
}

func retrieveParams(r *cc.ParameterList, i int) []funcArg {
	if r == nil {
		return nil
	}

	paramdecl := r.ParameterDeclaration
	if paramdecl != nil {
		// typename
		typeName := retrieveType(paramdecl.DeclarationSpecifiers)
		paramName := ""
		switch paramdecl.Case {
		case cc.ParameterDeclarationAbstract:
			decl := paramdecl.AbstractDeclarator

			if decl != nil {
				if decl.Case == cc.AbstractDeclaratorDecl &&
					decl.DirectAbstractDeclarator.Token.SrcStr() == "[" {
					typeName = typeName + "[]"
				}
				if decl.Case == cc.AbstractDeclaratorPtr {
					typeName = typeName + " *"
				}
			}
		case cc.ParameterDeclarationDecl:
			decl := paramdecl.Declarator
			paramName = decl.DirectDeclarator.Token.SrcStr()
			if decl.Pointer != nil && decl.Pointer.Case == cc.PointerTypeQual {
				typeName = typeName + " *"
			}
			if decl.DirectDeclarator.Case == cc.DirectDeclaratorArr {
				typeName = typeName + "[]"
				paramName = decl.DirectDeclarator.DirectDeclarator.Token.SrcStr()
			}
		}
		if paramName == "" {
			paramName = fmt.Sprintf("p%d", i)
		}
		rustname, dontUse := getRustParamType(typeName)
		return append([]funcArg{{
			name:     paramName,
			typeName: typeName,
			rustName: rustname,
			dontUse:  dontUse,
		}}, retrieveParams(r.ParameterList, i+1)...)
	}

	return retrieveParams(r.ParameterList, i+1)
}

func (flist *funcListInput) retrieveFuncDef(d *cc.ExternalDeclaration) *funcDef {
	if d == nil {
		return nil
	}

	if d.Declaration == nil {
		return nil
	}

	// DeclarationSpecifiers InitDeclaratorList AttributeSpecifierList ';'  // Case DeclarationDecl
	if d.Declaration.Case != cc.DeclarationDecl {
		return nil
	}

	if d.Declaration.InitDeclaratorList == nil {
		return nil
	}

	if d.Declaration.InitDeclaratorList.InitDeclarator == nil {
		return nil
	}

	//	InitDeclarator:
	//	        Declarator Asm                  // Case InitDeclaratorDecl
	//	|       Declarator Asm '=' Initializer  // Case InitDeclaratorInit
	if d.Declaration.InitDeclaratorList.InitDeclarator.Case != cc.InitDeclaratorDecl {
		return nil
	}

	decl := d.Declaration.InitDeclaratorList.InitDeclarator.Declarator.DirectDeclarator

	if decl == nil {
		return nil
	}

	// function name
	if decl.DirectDeclarator == nil || decl.DirectDeclarator.Case != cc.DirectDeclaratorIdent {
		return nil
	}

	name := decl.DirectDeclarator.Token.SrcStr()

	is32, is64, betterName := flist.findFunc(name)

	if !is32 && !is64 {
		return nil
	}

	returnType := retrieveType(d.Declaration.DeclarationSpecifiers)

	// retrieve arguments
	fdef := funcDef{
		RawName:    name,
		ReturnType: returnType,
		BetterName: betterName,
		args:       retrieveParams(decl.ParameterTypeList.ParameterList, 0),
		is32:       is32,
	}

	return &fdef
}

func run(cmd *cobra.Command, args []string) {
	if mklPath == "" {
		mklRoot := os.Getenv("MKLROOT")
		if mklRoot == "" {
			mklRoot = "/opt/intel/oneapi/mkl/latest"
		}
		mklPath = path.Join(mklRoot, "include", "mkl.h")
	}

	includePath := path.Dir(mklPath)

	compiler := getOrPanic(cc.NewConfig("", ""))
	compiler.IncludePaths = append(compiler.IncludePaths, includePath)

	ccast := getOrPanic(cc.Translate(compiler, []cc.Source{
		{Name: "<predefined>", Value: compiler.Predefined},
		{Name: "<builtin>", Value: cc.Builtin},
		{Name: mklPath},
	}))

	flist := readFuncList(inputFuncsPath)

	funcs := make([]funcDef, 0)

	cctu := ccast.TranslationUnit

	for thistu := cctu; thistu != nil; thistu = thistu.TranslationUnit {
		f := flist.retrieveFuncDef(thistu.ExternalDeclaration)
		if f != nil {
			funcs = append(funcs, *f)
		}
	}
	var b bytes.Buffer

	switch {
	case forC:
		ccTmpl := getOrPanic(template.New("cc-tmpl").Parse(ccTmplText))
		orPanic(ccTmpl.Execute(&b, &tmplInput{
			funcDefs:        funcs,
			providerCrate:   mklProviderCrate,
			DesiredFuncList: flist.desiredFuncList,
		}))
	case forGo:
		goTmpl := getOrPanic(template.New("go-tmpl").Parse(goTmplText))
		orPanic(goTmpl.Execute(&b, &tmplInput{
			funcDefs:        funcs,
			providerCrate:   mklProviderCrate,
			DesiredFuncList: flist.desiredFuncList,
		}))
		newb := getOrPanic(format.Source(b.Bytes(), format.Options{LangVersion: "1.22"}))
		b.Reset()
		getOrPanic(b.Write(newb))
	default:
		rsTmpl := getOrPanic(template.New("rs-tmpl").Parse(rsTmplText))
		orPanic(rsTmpl.Execute(&b, &tmplInput{
			funcDefs:        funcs,
			providerCrate:   mklProviderCrate,
			DesiredFuncList: flist.desiredFuncList,
		}))
	}

	orPanic(os.WriteFile(outputFile, b.Bytes(), 0o666))
}

var longDescription = `generate select mkl bindings for rust, c++, or go.

Use - for stdin, for example

  cat <<-EOF | go run github.com/fardream/gen-mkl-wrapper@main -i - -o mkl.rs
  v*Mul
  cblas_*gemm
  cblas_*gemv
  cblas_*syrk
  cblas_*swap
  LAPACKE_*potrs
  LAPACKE_*trtrs
  v*RngGaussian
  EOF
`

var crateLongDescription = `crate/module that provides the C bindings for MKL functions.
This can be some mod that contains the bindgen-ed wrappers for mkl.h file.
For example, if the generated bindings are in the same crate but under mod mkl_c, use "crate::mkl_c" as this parameter.
Add the generated bindings to the root mod of the crate to use default option "crate".
`

func main() {
	cmd := &cobra.Command{
		Short: "generate select bindings of MKL for rust, c++, or go",
		Use:   "gen-mkl-wrapper",
		Args:  cobra.NoArgs,
		Long:  longDescription,
	}

	cmd.Flags().StringVarP(&inputFuncsPath, "input", "i", inputFuncsPath, "list of functions to generate. use * for s/d, use # for S/D. use - for stdin.")
	cmd.MarkFlagFilename("input")
	cmd.MarkFlagRequired("input")

	cmd.Flags().StringVarP(&outputFile, "output", "o", outputFile, "output file")
	cmd.MarkFlagFilename("output", "rs", "h", "go")
	cmd.MarkFlagRequired("output")

	cmd.Flags().StringVarP(&mklPath, "mkl-header", "m", mklPath, "path to mkl.h file")
	cmd.MarkFlagFilename("mkl-header", ".h")

	cmd.Flags().StringVarP(&mklProviderCrate, "mkl-provider-crate", "c", mklProviderCrate, crateLongDescription)
	cmd.Flags().StringVarP(&traitName, "trait-name", "t", traitName, "trait name")

	cmd.Flags().BoolVar(&forC, "for-cc", forC, "output c++")

	cmd.Flags().BoolVar(&forGo, "for-go", forGo, "output go")
	cmd.Flags().StringVar(&goPackageName, "gopkg", goPackageName, "go package name")

	cmd.Run = run
	cmd.Execute()
}
