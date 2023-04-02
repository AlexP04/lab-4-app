mkdir -p ~/.streamlit/
echo "\
[theme]\n\
base=\"light\"\n\
primaryColor=\"#5fe0de\"\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
