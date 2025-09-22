# Financial domain prompts for hallucination detection
financial_prompts = [
    # Basic Financial Concepts
    {"question": "What is inflation?", "answer": "Inflation is the rate at which the general level of prices for goods and services rises"},
    {"question": "What is GDP?", "answer": "Gross Domestic Product (GDP) is the total monetary value of all finished goods and services produced within a country"},
    {"question": "What is a recession?", "answer": "A recession is a significant decline in economic activity lasting more than a few months"},
    {"question": "What is the Federal Reserve?", "answer": "The Federal Reserve is the central banking system of the United States"},
    {"question": "What is quantitative easing?", "answer": "Quantitative easing is a monetary policy where central banks buy securities to increase money supply"},
    
    # Market Terms
    {"question": "What is a bear market?", "answer": "A bear market is when stock prices fall 20% or more from recent highs"},
    {"question": "What is a bull market?", "answer": "A bull market is when stock prices rise 20% or more from recent lows"},
    {"question": "What is market capitalization?", "answer": "Market cap is the total value of all a company's shares of stock"},
    {"question": "What is a dividend?", "answer": "A dividend is a payment made by a corporation to its shareholders"},
    {"question": "What is a bond?", "answer": "A bond is a fixed-income security that represents a loan made by an investor to a borrower"},
    
    # Banking and Finance
    {"question": "What is the prime rate?", "answer": "The prime rate is the interest rate that commercial banks charge their most creditworthy customers"},
    {"question": "What is LIBOR?", "answer": "LIBOR (London Interbank Offered Rate) is a benchmark interest rate used in financial markets"},
    {"question": "What is a credit score?", "answer": "A credit score is a numerical representation of a person's creditworthiness"},
    {"question": "What is compound interest?", "answer": "Compound interest is interest calculated on the initial principal and accumulated interest"},
    {"question": "What is liquidity?", "answer": "Liquidity refers to how easily an asset can be converted to cash without affecting its price"},
    
    # Investment Terms
    {"question": "What is a mutual fund?", "answer": "A mutual fund is an investment vehicle that pools money from many investors to buy securities"},
    {"question": "What is an ETF?", "answer": "An ETF (Exchange-Traded Fund) is a type of investment fund traded on stock exchanges"},
    {"question": "What is diversification?", "answer": "Diversification is the practice of spreading investments across different assets to reduce risk"},
    {"question": "What is a hedge fund?", "answer": "A hedge fund is an investment partnership that uses advanced strategies to generate returns"},
    {"question": "What is a 401(k)?", "answer": "A 401(k) is a retirement savings plan sponsored by an employer"},
    
    # Economic Indicators
    {"question": "What is the unemployment rate?", "answer": "The unemployment rate is the percentage of the labor force that is unemployed and actively seeking work"},
    {"question": "What is the consumer price index?", "answer": "The CPI measures the average change in prices paid by consumers for goods and services"},
    {"question": "What is the producer price index?", "answer": "The PPI measures the average change in selling prices received by domestic producers"},
    {"question": "What is the trade deficit?", "answer": "A trade deficit occurs when a country imports more goods and services than it exports"},
    {"question": "What is the national debt?", "answer": "The national debt is the total amount of money that a government owes to creditors"},
    
    # Currency and Exchange
    {"question": "What is a currency exchange rate?", "answer": "An exchange rate is the rate at which one currency can be exchanged for another"},
    {"question": "What is the US dollar index?", "answer": "The US Dollar Index measures the value of the US dollar against a basket of foreign currencies"},
    {"question": "What is cryptocurrency?", "answer": "Cryptocurrency is a digital or virtual currency secured by cryptography"},
    {"question": "What is Bitcoin?", "answer": "Bitcoin is the first and largest cryptocurrency by market capitalization"},
    {"question": "What is blockchain?", "answer": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records"},
    
    # Risk Management
    {"question": "What is risk management?", "answer": "Risk management is the process of identifying, assessing, and controlling threats to an organization"},
    {"question": "What is hedging?", "answer": "Hedging is an investment strategy used to reduce the risk of adverse price movements"},
    {"question": "What is value at risk?", "answer": "Value at Risk (VaR) is a statistical measure of the risk of loss for investments"},
    {"question": "What is systematic risk?", "answer": "Systematic risk is the risk inherent to the entire market or market segment"},
    {"question": "What is unsystematic risk?", "answer": "Unsystematic risk is the risk specific to a particular company or industry"},
    
    # Regulatory and Compliance
    {"question": "What is the SEC?", "answer": "The Securities and Exchange Commission (SEC) regulates the securities markets in the US"},
    {"question": "What is the FDIC?", "answer": "The Federal Deposit Insurance Corporation (FDIC) insures deposits in US banks"},
    {"question": "What is Basel III?", "answer": "Basel III is a set of international banking regulations developed after the 2008 financial crisis"},
    {"question": "What is Dodd-Frank?", "answer": "Dodd-Frank is a comprehensive financial reform law passed in 2010"},
    {"question": "What is Sarbanes-Oxley?", "answer": "Sarbanes-Oxley (SOX) is a law that sets requirements for public company boards and accounting firms"},
    
    # Advanced Financial Concepts
    {"question": "What is a derivative?", "answer": "A derivative is a financial contract whose value is derived from an underlying asset"},
    {"question": "What is an option?", "answer": "An option is a contract that gives the holder the right to buy or sell an asset at a specific price"},
    {"question": "What is a futures contract?", "answer": "A futures contract is an agreement to buy or sell an asset at a predetermined future date and price"},
    {"question": "What is arbitrage?", "answer": "Arbitrage is the practice of taking advantage of price differences in different markets"},
    {"question": "What is leverage?", "answer": "Leverage is the use of borrowed money to increase the potential return of an investment"},
    
    # International Finance
    {"question": "What is the IMF?", "answer": "The International Monetary Fund (IMF) is an international organization that promotes global monetary cooperation"},
    {"question": "What is the World Bank?", "answer": "The World Bank is an international financial institution that provides loans to developing countries"},
    {"question": "What is the G7?", "answer": "The G7 is a group of seven major advanced economies"},
    {"question": "What is the G20?", "answer": "The G20 is a group of 20 major economies that meets to discuss global economic issues"},
    {"question": "What is the WTO?", "answer": "The World Trade Organization (WTO) is an international organization that regulates international trade"},
    
    # Corporate Finance
    {"question": "What is a balance sheet?", "answer": "A balance sheet is a financial statement that shows a company's assets, liabilities, and equity"},
    {"question": "What is an income statement?", "answer": "An income statement shows a company's revenues, expenses, and profits over a period"},
    {"question": "What is a cash flow statement?", "answer": "A cash flow statement shows how changes in balance sheet accounts affect cash"},
    {"question": "What is EBITDA?", "answer": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization"},
    {"question": "What is working capital?", "answer": "Working capital is the difference between a company's current assets and current liabilities"},
    
    # Personal Finance
    {"question": "What is a budget?", "answer": "A budget is a plan for managing income and expenses over a specific period"},
    {"question": "What is an emergency fund?", "answer": "An emergency fund is money set aside to cover unexpected expenses"},
    {"question": "What is a credit card?", "answer": "A credit card is a payment card that allows the holder to borrow money to make purchases"},
    {"question": "What is a mortgage?", "answer": "A mortgage is a loan used to purchase real estate, secured by the property itself"},
    {"question": "What is insurance?", "answer": "Insurance is a contract that provides financial protection against specific risks"},
    
    # Economic Theories
    {"question": "What is Keynesian economics?", "answer": "Keynesian economics is a theory that advocates for government intervention in the economy"},
    {"question": "What is supply and demand?", "answer": "Supply and demand is an economic model that explains price determination in a market"},
    {"question": "What is the invisible hand?", "answer": "The invisible hand is Adam Smith's concept that individuals' self-interest benefits society"},
    {"question": "What is comparative advantage?", "answer": "Comparative advantage is the ability to produce goods at a lower opportunity cost than others"},
    {"question": "What is the multiplier effect?", "answer": "The multiplier effect is the proportional increase in final income from an injection of spending"},
    
    # Financial Crises and Events
    {"question": "What was the Great Depression?", "answer": "The Great Depression was a severe worldwide economic depression in the 1930s"},
    {"question": "What was the 2008 financial crisis?", "answer": "The 2008 financial crisis was a global economic downturn triggered by the collapse of the housing market"},
    {"question": "What was the dot-com bubble?", "answer": "The dot-com bubble was a stock market bubble in the late 1990s involving internet companies"},
    {"question": "What was the Asian financial crisis?", "answer": "The Asian financial crisis was a period of financial crisis that affected Asian markets in 1997"},
    {"question": "What was the savings and loan crisis?", "answer": "The savings and loan crisis was the failure of 1,043 out of 3,234 savings and loan associations"},
    
    # Modern Financial Technology
    {"question": "What is fintech?", "answer": "Fintech is technology used to support or enable banking and financial services"},
    {"question": "What is robo-advisory?", "answer": "Robo-advisory is digital platforms that provide automated investment advice"},
    {"question": "What is mobile banking?", "answer": "Mobile banking is the use of smartphones to conduct banking transactions"},
    {"question": "What is contactless payment?", "answer": "Contactless payment is a secure method for consumers to purchase products using a debit or credit card"},
    {"question": "What is open banking?", "answer": "Open banking is a banking practice that provides third-party access to financial data"},
    
    # Environmental and Social Finance
    {"question": "What is ESG investing?", "answer": "ESG investing considers environmental, social, and governance factors in investment decisions"},
    {"question": "What is impact investing?", "answer": "Impact investing aims to generate positive social and environmental impact alongside financial returns"},
    {"question": "What is green finance?", "answer": "Green finance refers to financial investments flowing into sustainable development projects"},
    {"question": "What is social impact bonds?", "answer": "Social impact bonds are contracts with the public sector where investors fund social programs"},
    {"question": "What is carbon trading?", "answer": "Carbon trading is a market-based approach to controlling pollution by providing economic incentives"},
    
    # Behavioral Finance
    {"question": "What is behavioral finance?", "answer": "Behavioral finance studies how psychological factors affect financial decision-making"},
    {"question": "What is loss aversion?", "answer": "Loss aversion is the tendency to prefer avoiding losses to acquiring equivalent gains"},
    {"question": "What is confirmation bias?", "answer": "Confirmation bias is the tendency to search for information that confirms one's preconceptions"},
    {"question": "What is herd behavior?", "answer": "Herd behavior is the tendency for individuals to follow the actions of a larger group"},
    {"question": "What is anchoring bias?", "answer": "Anchoring bias is the tendency to rely too heavily on the first piece of information encountered"},
    
    # Quantitative Finance
    {"question": "What is quantitative finance?", "answer": "Quantitative finance uses mathematical models and computational methods to analyze financial markets"},
    {"question": "What is algorithmic trading?", "answer": "Algorithmic trading uses computer programs to execute trades based on predefined instructions"},
    {"question": "What is high-frequency trading?", "answer": "High-frequency trading uses powerful computers to execute trades in fractions of a second"},
    {"question": "What is machine learning in finance?", "answer": "Machine learning in finance uses algorithms to analyze data and make predictions about markets"},
    {"question": "What is risk modeling?", "answer": "Risk modeling uses mathematical techniques to assess the potential for losses in investments"},
    
    # Real Estate Finance
    {"question": "What is real estate investment?", "answer": "Real estate investment involves purchasing property to generate income or appreciation"},
    {"question": "What is a REIT?", "answer": "A REIT (Real Estate Investment Trust) is a company that owns and operates income-producing real estate"},
    {"question": "What is property valuation?", "answer": "Property valuation is the process of determining the economic value of real estate"},
    {"question": "What is a cap rate?", "answer": "A cap rate is the ratio of net operating income to property asset value"},
    {"question": "What is commercial real estate?", "answer": "Commercial real estate is property used for business purposes rather than residential living"},
    
    # Insurance and Risk
    {"question": "What is actuarial science?", "answer": "Actuarial science is the discipline that applies mathematical and statistical methods to assess risk"},
    {"question": "What is underwriting?", "answer": "Underwriting is the process of evaluating the risk of insuring a person or asset"},
    {"question": "What is a premium?", "answer": "A premium is the amount paid for an insurance policy"},
    {"question": "What is a deductible?", "answer": "A deductible is the amount paid out of pocket before insurance coverage begins"},
    {"question": "What is reinsurance?", "answer": "Reinsurance is insurance for insurance companies to protect against large losses"},
    
    # Financial Planning
    {"question": "What is financial planning?", "answer": "Financial planning is the process of creating a strategy to manage finances and achieve goals"},
    {"question": "What is asset allocation?", "answer": "Asset allocation is the process of dividing investments among different asset categories"},
    {"question": "What is dollar-cost averaging?", "answer": "Dollar-cost averaging is investing a fixed amount regularly regardless of market conditions"},
    {"question": "What is rebalancing?", "answer": "Rebalancing is the process of realigning the weightings of a portfolio's assets"},
    {"question": "What is tax planning?", "answer": "Tax planning is the analysis of a financial situation to minimize tax liability"},
    
    # International Trade
    {"question": "What is free trade?", "answer": "Free trade is international trade without government restrictions or barriers"},
    {"question": "What is a tariff?", "answer": "A tariff is a tax imposed on imported goods and services"},
    {"question": "What is a trade war?", "answer": "A trade war is an economic conflict between countries using trade barriers"},
    {"question": "What is globalization?", "answer": "Globalization is the process of international integration arising from the interchange of world views"},
    {"question": "What is outsourcing?", "answer": "Outsourcing is the practice of contracting out business processes to external service providers"},
    
    # Financial Instruments
    {"question": "What is a stock?", "answer": "A stock is a type of security that represents ownership in a corporation"},
    {"question": "What is a commodity?", "answer": "A commodity is a basic good used in commerce that is interchangeable with other goods"},
    {"question": "What is a swap?", "answer": "A swap is a derivative contract in which two parties exchange financial instruments"},
    {"question": "What is a warrant?", "answer": "A warrant is a derivative that gives the holder the right to buy securities at a specific price"},
    {"question": "What is a convertible bond?", "answer": "A convertible bond is a type of bond that can be converted into shares of stock"},
    
    # Economic Indicators and Data
    {"question": "What is the yield curve?", "answer": "The yield curve is a graph showing the relationship between bond yields and maturity dates"},
    {"question": "What is the VIX?", "answer": "The VIX (Volatility Index) measures market expectations of volatility"},
    {"question": "What is the S&P 500?", "answer": "The S&P 500 is a stock market index measuring the performance of 500 large companies"},
    {"question": "What is the Dow Jones?", "answer": "The Dow Jones Industrial Average is a stock market index of 30 large companies"},
    {"question": "What is the NASDAQ?", "answer": "The NASDAQ is a global electronic marketplace for buying and selling securities"},
    
    # Financial Regulations
    {"question": "What is the CFPB?", "answer": "The Consumer Financial Protection Bureau (CFPB) is a US government agency that protects consumers"},
    {"question": "What is the CFTC?", "answer": "The Commodity Futures Trading Commission (CFTC) regulates US derivatives markets"},
    {"question": "What is the OCC?", "answer": "The Office of the Comptroller of the Currency (OCC) charters and regulates national banks"},
    {"question": "What is the CFPB?", "answer": "The Consumer Financial Protection Bureau (CFPB) is a US government agency that protects consumers"},
    {"question": "What is the CFPB?", "answer": "The Consumer Financial Protection Bureau (CFPB) is a US government agency that protects consumers"},
]
