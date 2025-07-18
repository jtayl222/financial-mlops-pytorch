#!/bin/bash

kubectl api-resources --verbs=list --namespaced -o name | grep seldon | xargs -n 1 kubectl -n financial-mlops-pytorch get --ignore-not-found
