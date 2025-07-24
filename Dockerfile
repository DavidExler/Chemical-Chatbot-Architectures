FROM python:3.13

RUN pip install numpy rdkit pubchempy mordred sympy molmass[all] ase
