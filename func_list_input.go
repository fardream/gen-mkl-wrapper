package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

type funcListInput struct {
	forFloat64      map[string]string
	forFloat32      map[string]string
	desiredFuncList []string
}

func splitName(v string, sep string, s64 string, s32 string) (betterName string, n64 string, n32 string) {
	fixes := strings.Split(v, sep)
	if len(fixes) != 2 {
		log.Panicf("%s doesn't containt a valid name", v)
	}
	betterName = strings.Join(fixes, "")
	n64 = fmt.Sprintf("%s%s%s", fixes[0], s64, fixes[1])
	n32 = fmt.Sprintf("%s%s%s", fixes[0], s32, fixes[1])
	return
}

func readFuncList(input string) *funcListInput {
	f := &funcListInput{
		forFloat64: make(map[string]string),
		forFloat32: make(map[string]string),
	}

	content := ""
	if input == "-" {
		content = string(getOrPanic(io.ReadAll(os.Stdin)))
	} else {
		content = string(getOrPanic(os.ReadFile(input)))
	}

	for _, inputline := range strings.Split(content, "\n") {
		v := strings.TrimRight(strings.TrimLeft(inputline, " "), " ")
		if v == "" {
			continue
		}

		f.desiredFuncList = append(f.desiredFuncList, v)

		var bn, n64, n32 string
		if strings.Contains(v, "*") {
			bn, n64, n32 = splitName(v, "*", "d", "s")
		} else if strings.Contains(v, "#") {
			bn, n64, n32 = splitName(v, "#", "D", "S")
		}

		f.forFloat64[n64] = bn
		f.forFloat32[n32] = bn

	}

	return f
}

func (f *funcListInput) findFunc(funcName string) (is32 bool, is64 bool, betterName string) {
	betterName, is32 = f.forFloat32[funcName]
	if is32 {
		is64 = false
		return
	}
	betterName, is64 = f.forFloat64[funcName]
	if is64 {
		is32 = false
		return
	}

	return
}
