
v0.0.5 / 2017-09-29
===================

  * changed version numbering scheme
  * removed unused directories
  * incorporate project structure upgrades
  * fixed pkg name typo

v2017.09.29.1
=============

  * data.validation: added `is_in` and `not_in`
  * added misc.py
  * formatting
  * added data.parsers.bch.biorepository.py
  * data.parsers: renamed and made pkg
  * load_recode.build_generic_table: drop_duplicates before returning

v2017.09.14.1
=============

  * Biorepository field non_bch_pt to bch_pt
  * rc2np.text = str not np.object
  * added schema support to BaseData
  * style and formatting
  * Refactored Data Classes to inherit from database agnostic BaseData

v2017.03.30.1
=============

  * Added History.md
  * Merge branch 'develop'
  * Finished RegistryRedCapData()
  * added asset_intake.py
  * Merge branch 'develop' into feature/registry
  * Merge branch 'feature/biorepo' into develop
  * code to build the SQL-like tables for the BIOREPO
  * Merge branch 'feature/registry' into develop
  * largish update: most table building funcs operational with automatic casting
  * updates to RedCapData
  * added package for defining the SQL tables in sqlalchemy
  * added sqlalchemy to reqs
  * Merge branch 'develop'
  * Merge branch 'develop' into feature/registry
  * Merge branch 'feature/oo_api' into develop
  * load_recode OOP api is operational obviating the need for separate ones
  * Separated RedCapData classes and support funcs to new load_recode.py
  * RegistryRedCapData() works
  * removed cached notebooks and reports
  * notebook update
  * added/integrated make_redcap_validation_table() and now use pd.Timestamp
  * cleaned up crf_dag.py
  * ignore notebook and reports
  * notebook updates
  * HREG branching logic, and choice_defintion parsers
  * added lxml as req
  * added a few graphml versions of the DOT files
  * begun adding pyparsing parser for branch logic
  * fixed load_recode/redcap_dump import statements
  * Merge branch 'feature/build_erd' into develop
  * updated ignore: /data
  * notebook and reference notes update
  * Split load_recode and redcap_dump into biorepo and hreg prefixed versions
  * Merged feature/registry into develop
  * Added separate folders to build the ERDs in Notes
  * Merged develop into master
  * Merged develop into feature/registry
  * changed the way we deal with ignored files in /data/*
  * Change import statement for load_recode (not sure why that was needed)
  * Initial attempts are mixed...
  * separating before adding feature/validate_with_engarde
