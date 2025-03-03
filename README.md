

### This repo is my approach on the 'Zindi Fraud Detection in Electricity and Gas Consumption Challenge'

The raw data can be downloaded from Zindi. The modified data and result is provided with this repo.  
This repo explores the datasets in the 'EDA.ipynb', subsequently trains a XGBoost model in 'modeling_XGB.ipynb' and finally applies the model on the real test data in 'Testdata_submission.ipynb'.


The following text is the original README from Zindi:   



# Fraud_Detection_zindi
Fraud Detection in Electricity and Gas Consumption Challenge from Zindi  
https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge

## Description (from zindi.africa)
The Tunisian Company of Electricity and Gas (STEG) is a public and a non-administrative company, it is responsible for delivering electricity and gas across Tunisia. The company suffered tremendous losses in the order of 200 million Tunisian Dinars due to fraudulent manipulations of meters by consumers.

Using the client’s billing history, the aim of the challenge is to detect and recognize clients involved in fraudulent activities.

The solution will enhance the company’s revenues and reduce the losses caused by such fraudulent activities.

About STEG (https://www.steg.com.tn/en/institutionnel/mission.html)


The Tunisian Company of Electricity and Gas (STEG) is a public and a non-administrative company. It is responsible for delivering electricity and gas across Tunisia.

Evaluation
The metric used for this challenge is Area Under the Curve.

Then the submission file should be as follows:

client_id       target
test_Client_0   0.986
test_Client_1   0.011
test_Client_10  0.734

## Data (from zindi.africa)

The data provided by STEG is composed of two files. The first one is comprised of client data and the second one contains billing history from 2005 to 2019.

There are 2 .zip files for download, train.zip, and test.zip and a SampleSubmission.csv. In each .zip file you will find a client and invoice file.

Variable definitions

**Client:**
- Client_id: Unique id for client
- District: District where the client is
- Client_catg: Category client belongs to
- Region: Area where the client is
- Creation_date: Date client joined
- Target: fraud:1 , not fraud: 0    

  
**Invoice data:**   
- Client_id: Unique id for the client  
- Invoice_date: Date of the invoice  
- Tarif_type: Type of tax  
- Counter_number:  
- Counter_statue: takes up to 5 values such as working fine, not   -working, on hold statue, ect  
- Counter_code:  
- Reading_remarque: notes that the STEG agent takes during his visit to the client (e.g: If the counter shows something wrong, the agent gives a bad score)  
- Counter_coefficient: An additional coefficient to be added when standard consumption is exceeded  
- Consommation_level_1: Consumption_level_1  
- Consommation_level_2: Consumption_level_2  
- Consommation_level_3: Consumption_level_3  
- Consommation_level_4: Consumption_level_4  
- Old_index: Old index  
- New_index: New index  
- Months_number: Month number  
- Counter_type: Type of counter  