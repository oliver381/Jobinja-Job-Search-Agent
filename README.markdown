# Jobinja Job Search Agent

## Description

The **Jobinja Job Search Agent** is a powerful Python application designed to simplify job searches on [Jobinja.ir](https://jobinja.ir), a leading job portal in Iran. This tool allows users to input job tags and skills, generates a relevant job title using a fine-tuned [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) model, creates a targeted search keyword with [Ollama](https://ollama.com), and automates the search process on Jobinja using [Selenium](https://www.selenium.dev). Results are displayed in a user-friendly, right-to-left (RTL) web interface built with [Streamlit](https://streamlit.io), tailored for Persian-speaking users. This project is ideal for job seekers, HR professionals, and developers interested in AI-driven job search automation.

## Why Use This Project?

- **Streamlined Job Search**: Automates the process of finding relevant job listings, saving time for users.
- **AI-Powered Title Generation**: Generates precise job titles using a fine-tuned DeepSeek model, optimized for Persian job markets.
- **Targeted Search Keywords**: Creates concise, market-specific search keywords using Ollama LLM.
- **User-Friendly Interface**: Offers an intuitive Streamlit interface with RTL support for Persian users.
- **Open-Source**: Encourages community contributions to enhance functionality and adapt to new use cases.

## Features

- **Job Title Generation**: Uses a fine-tuned DeepSeek model to suggest job titles based on user-provided tags and skills.
- **Search Keyword Generation**: Employs Ollama’s `qwen2.5:latest` model to create concise, Persian search keywords.
- **Automated Job Search**: Scrapes up to 10 job listings from Jobinja.ir using Selenium, including titles, companies, locations, and links.
- **RTL Interface**: Displays results in a clean, Persian-friendly Streamlit interface with custom CSS styling.
- **Robust Error Handling**: Manages errors during title generation, keyword creation, and web scraping, with user feedback.
- **Workflow Management**: Utilizes [LangGraph](https://langchain-ai.github.io/langgraph/) to orchestrate the process from input to results.

## Installation

### Prerequisites

- **Python 3.8 or Higher**: Download from [Python.org](https://www.python.org/downloads/).
- **ChromeDriver**: Download from [ChromeDriver](https://chromedriver.chromium.org/downloads) and add to your system PATH or place in the project directory.
- **Ollama**: Install from [Ollama.com](https://ollama.com) and pull the `qwen2.5` model:
  ```bash
  ollama pull qwen2.5
  ollama serve
  ```
- **Fine-Tuned Model**: Ensure the fine-tuned DeepSeek model is available in the `./jobinja_model` directory. This model should include either full model files (`config.json`, `pytorch_model.bin`, tokenizer files) or PEFT adapter files (`adapter_config.json`, `adapter_model.bin`).

### Install Dependencies

Install the required Python libraries:
```bash
pip install streamlit selenium transformers peft torch langchain-ollama langgraph
```

### System Requirements

- **Internet Connection**: Required for initial setup and web scraping.
- **Hardware**: 16GB RAM and a GPU (e.g., NVIDIA T4) recommended for model inference, though CPU is supported.
- **Operating System**: Compatible with Windows, macOS, or Linux.
- **Browser**: Google Chrome with a matching ChromeDriver version.

### Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/armanjscript/Jobinja-Job-Search-Agent.git
   cd Jobinja-Job-Search-Agent
   ```

2. **Place Model Files**:
   - Copy the fine-tuned model files to the `./jobinja_model` directory. If you don’t have the model, refer to the fine-tuning instructions in `FineTuning_Jobinja.ipynb` (not included here but assumed to be part of the project).

3. **Verify ChromeDriver**:
   - Ensure ChromeDriver matches your Chrome browser version and is accessible in your PATH.

4. **Start Ollama**:
   ```bash
   ollama serve
   ```

## Usage

1. **Launch the Application**:
   ```bash
   streamlit run main.py
   ```
   - This opens the app in your default web browser.

2. **Interact with the Interface**:
   - In the Streamlit interface, enter job tags (e.g., "برنامه‌نویسی, توسعه وب, پایتون") and skills (e.g., "Django, Flask, HTML, CSS") in the provided text areas.
   - Click the "جستجوی هوشمند" (Smart Search) button to initiate the process.

3. **View Results**:
   - The app displays:
     - **Generated Job Title**: The AI-generated job title based on your inputs.
     - **Search Keyword**: A concise keyword used for the Jobinja search.
     - **Job Listings**: Up to 10 job listings with titles, companies, locations, and links to Jobinja.ir.

4. **Troubleshooting**:
   - **Model Loading Errors**: Ensure the `./jobinja_model` directory contains valid model files. Check the troubleshooting guide in `jobinja_title_generator.py`.
   - **Selenium Issues**: Verify ChromeDriver compatibility and internet connectivity.
   - **Ollama Errors**: Confirm Ollama is running and the `qwen2.5` model is pulled.

## Technologies Used

| Technology | Role |
|------------|------|
| **Python** | Primary programming language. |
| **Streamlit** | Creates the RTL web interface for user interaction. |
| **Selenium** | Automates job searches on Jobinja.ir. |
| **Hugging Face Transformers** | Loads and uses the fine-tuned DeepSeek model for title generation. |
| **PEFT** | Supports efficient loading of fine-tuned models with LoRA adapters. |
| **Torch** | Handles model inference with GPU/CPU support. |
| **LangChain-Ollama** | Generates search keywords using the Ollama LLM. |
| **LangGraph** | Manages the workflow from title generation to job search. |

## Example Output

For inputs:
- **Tags**: "فروش و بازاریابی"
- **Skills**: "فروش تلفنی, اصول و فنون مذاکره"
- **Generated Title**: "استخدام کارشناس فروش تلفنی"
- **Search Keyword**: "کارشناس فروش"
- **Job Listings**: Up to 10 results, e.g., "کارشناس فروش تلفنی - شرکت XYZ - تهران - [Link]"

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository on [GitHub](https://github.com/armanjscript/Jobinja-Job-Search-Agent).
2. Create a new branch for your changes.
3. Submit a pull request with a detailed description.
4. For bug reports or feature requests, open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, open an issue on [GitHub](https://github.com/armanjscript/Jobinja-Job-Search-Agent) or email [armannew73@gmail.com].

## Potential Enhancements

- Add support for filtering job listings by location or salary.
- Integrate additional job portals for broader search capabilities.
- Enhance the interface with visualizations of job trends.
- Optimize model inference for lower-end hardware using quantization.

#هوش_مصنوعی #فاین_تیونینگ #اسکریپینگ_وب #پروژه_پایتون #مدل_زبانی #Jobinja #Selenium #HuggingFace #Transformers #LoRA #PEFT #ایران_تک #فناوری #AI_در_ایران
