mkdir -p ~/.streamlit/
echo "\
[theme]\n\
base=\"light\"\n\
primaryColor=\"#5fe0de\"
[server]
port = $PORT
enableCORS = false 
headless = true
\n\
" > ~/.streamlit/config.toml
