# ComputationalPhysicsProject

## Initial Setup

* You only need to do this part one time
* Be at the base directory of the repository
* Run ```python3 -m venv Packages```
    * This sets up a virtual environment to install your libraries in
* Run ```source Packages/bin/activate```
    * This spins up the virtual environment
* Run ```pip install -r /path/to/requirements.txt```
    * This downloads the libraries that we will use for the project. Currently, this is just
        * numpy
        * scipy
        * matplotlib

## Recurring Setup

* After you set things up initially, you only need to run the following command from the base of the repo:
```

source Packages/bin/activate

```

* Make sure that you exit the virtual environment if you are doing other work. This can be done with
```
    deactivate
```
