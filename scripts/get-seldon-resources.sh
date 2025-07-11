#!/bin/bash

kubectl api-resources --verbs=list --namespaced -o name | grep seldon | xargs -n 1 kubectl -n financial-inference get --ignore-not-found
