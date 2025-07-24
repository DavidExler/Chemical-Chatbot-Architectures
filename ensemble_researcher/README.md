# [Ensemble]

## Overview

The Ensemble Researcher architecture relies on multiple LLMs. A smaller model (student) generates rich context using a high temperature coefficient (adjust for optimal results) based on the output of the databases. A larger LLM writes queueries for the databases, waits for the students and then answers the users question based on the highly variable context provided by the student.
As the smaller model can easily be parrallelized, increasing the number of LLM calls does not increase the runtime.

## Architecture Diagram

![Architecture Diagram](flowchart_ensemble_researcher.png)