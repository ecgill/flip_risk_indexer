# Flip Risk Indexer

Emily Gill

Galvanize Data Science Immersive - January 2018

[FlipRiskIndexer.com](http://www.flipriskindexer.com)

## Table of Contents
1. [Background](#background)
2. [Web Application](#web-application)
3. [Data Collection and Cleaning](#data-collection-and-cleaning)
4. [Technologies Used](#technologies-used)
5. [Flip Risk Indexer](#flip-risk-indexer)
6. [Future Steps](#future-steps)
7. [Repo Structure](#repo-structure)

## Background

Privy is a real estate investment software that finds and analyzes deals fast. It informs agents of where investors are doing deals and it allows investors to make deals without needing an agent. In the world of Privy, investments come in three categories: fix 'n flips, pop-tops, and total tear downs.

In a market like Denver's, investment opportunities are a dime a dozen. This project will help Privy locate how hot zones in Denver are shifting, and use past investment opportunities to develop a Flip Risk Indexer for houses currently on the market.

## Web Application

I will create an app called FlipRiskIndexer using Flask and self-host it on AWS.

## Data Collection and Cleaning

I am currently working with MLS (Multiple Listing Services) real estate listings for Denver, a .csv obtained from Privy of 500K rows of current and previous real estate listings. Investment purchases are identified in three categories: fix 'n flips, pop-tops, and scrapes. A fix 'n flip is ID'd in the MLS as a property that was bought and then resold in a relatively short period of time with a price increase. A pop-top is ID'd in the MLS when a  property changes how many levels there are (i.e. it is purchased as a 2-story and then re-sold as a 3-story). Lastly, a scrap is ID'd in the MLS as a change in the year the house was build (e.g. it is purchased as a house  built in 1950 and then resold as a house built in 2017).

I read the .csv in as a pandas DataFrame, and cleaned the dataset a bit. The delimiting was off, so some rows were longer than others. I used the csv package to read in the .csv line by line, locate the inconsistencies and then used various ways to clean them up. Once clean, I categorized these three investment options, finding about 15K in total since 2008.

...more to come.

## Technologies Used

TBD

## Flip Risk Indexer

TBD

## Future Steps

TBD

## Repo structure
```
├── data (will contain csv files and pickles of data)
├── notebooks (will contain scripts used for EDA and visualizations)
├── src (will contain the scripts used to perform the analysis)
├── web_app
|     ├── static
|     ├── templates
|     └── app.py (will run the web app via Flask)
└── README.md
```
