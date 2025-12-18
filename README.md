# AWS ML Data Pipeline (Local + AWS Execution)



This repository demonstrates a reproducible data preprocessing pipeline designed to run locally for development/testing and in AWS for scalable execution.

Status: Initial scaffold committed.





Goal: this ML pipeline is intended to provide analytical AI capabilities on Non profit organizations.



\## Data Overview



This project uses IRS Publication 78 (Pub 78) bulk data as the authoritative

source for determining U.S. nonprofit organizations eligible to receive

tax-deductible charitable contributions.



https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads





\### Raw Data

The full Pub 78 dataset (August 2025 snapshot) is downloaded externally and is

not committed to this repository.This design supports reproducibility while avoiding versioning large,

regeneratable datasets in source control.



\### Samples

A small representative sample of the Pub 78 (500 NPOs) data is included in:



    data/samples/



This sample is used for local testing, schema validation, and pipeline

development.



\- \*\*EIN\*\* (Employer Identification Number)  

\- \*\*Organization Name\*\*  

\- \*\*City, State, Country\*\*  

\- \*\*IRC Subsection Classification Code\*\* (the tax-exempt category under which the organization is recognized)

