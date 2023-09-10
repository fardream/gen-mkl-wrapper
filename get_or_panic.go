package main

import "log"

func orPanic(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func getOrPanic[T any](in T, err error) T {
	orPanic(err)
	return in
}
