
these_does_not_work() {
  curl -o NASDAQ.csv 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download'
  curl -o AMEX.csv 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'
  curl -o NYSE.csv 'http://www.asx.com.au/asx/research/ASXListedCompanies.csv'
}

# Details on format http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
download_symbols() {
  curl -o nasdaqlisted.txt 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt' 
  curl -o otherlisted.txt 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt' 
}

download_symbols

