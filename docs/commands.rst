Commands
========

The Makefile contains the central entry points for common tasks related to this project.



* `make github_remote` launches a script to create and syncronize a remote github repository for this project.

* `make environment` uses the project's `requirements.txt` file to create and provision a conda environment for this project. Then it registers the conda environment with your local jupyter set up as an ipython kernel with the same name as the conda environment.

* `make clean_env` undoes everything that `make environment` sets up.

* `make serve_nb` starts a jupyter notebook with this project's `notebooks` directory as the root.

* `make clean_bytecode` removes all `__pycache__` directories and leftover `*.pyc` files from the project.







Syncing data to S3
^^^^^^^^^^^^^^^^^^

* `make sync_data_to_s3` will use `s3cmd` to recursively sync files in `data/` up to `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/`.
* `make sync_data_from_s3` will use `s3cmd` to recursively sync files from `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/` to `data/`.
