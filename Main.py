import pandas as pd

import LogisticRegression as log_reg

def main():
    df = pd.read_csv('Data/fake_job_postings.csv')

    log_reg.compute(df)

    # Preprocessing
if __name__ == '__main__':
    main()