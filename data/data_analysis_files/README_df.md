The .parquet files to use are:
*With ALL metadata* (first including 2026, second excluding 2026):
- `data/data_analysis_files/df_all_final_86pct_grades_09042026.parquet`
- `data/data_analysis_files/df_all_final_86pct_grades_excl_2026_09042026.parquet`

*For those where PDF is available* (first including 2026, second excluding 2026):
- `data/data_analysis_files/df_filtered_final_86pct_grades_09042026.parquet`
- `data/data_analysis_files/df_filtered_final_86pct_grades_excl_2026_09042026.parquet`

**The filese are located in the folder:** data/data_analysis_files/

**To load a file in for analysis:**
IMPORT_PATH = "../../data/data_analysis_files/"
FILE_FILTERED_GRADES = "df_filtered_final_86pct_grades_excl_2026_09042026.parquet"

def load_parquet_to_df(parquet_path, na=False):
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Successfully loaded Parquet from {parquet_path}")
        print(f"DataFrame shape: {df.shape}")
        if na:
            print(f"DataFrame N/A counts:\n{df.isna().sum()}\n")
        print(f"DataFrame columns: {df.columns.tolist()}\n")
        return df
    except Exception as e:
        print(f"Error loading Parquet from {parquet_path}: {e}")
        return None

df_filtered_final = load_parquet_to_df(IMPORT_PATH + FILE_FILTERED_GRADES)

**COLUMNS RELEVANT FOR ANALYSIS:**
- **`Publication Year`**: The year of publication
- **`MASTER THESIS TITLE`**: The english title of the thesis
- **`BY`**: The author(s) in the format "lastname, name" (if multiple auhtors, they're separated wiht ";") 
- **`SUPERVISED BY`**: The supervisor(s) in the format "lastname, name" (if multiple supervisors, they're separated with ";")
- **`num_tot_pages`**: Number of total pages in .pdf file
- **`num_cont_pages`**: Number of content pages in the .pdf file (excluding appendix, references etc.)
- **`handin_month`**: The month of handin exstracted from the .pdf file. *OBS(!):* disregard the year in the stirng, and use the metric `Publication Year` for true year.
- **`num_figures`**: Number of figures in the .pdf file
- **`num_tables`**: Number of tables in the .pdf file
- **`num_references`** Number of references listed in the section regarding bibliography in the .pdf file
- **`equation_count`**: Number of equations in the .pdf file
- **`total_sentences`**: Number of sentences in main content of .pdf file
- **`total_words`**: Number of words in main content of .pdf file
- **`unique_words`**: Number of unique words in main content of .pdf file
- **`avg_sentence_length`**: Average sentence lenght of main content of .pdf file
- **`avg_word_length`**: Average word lenght of main conent of .pdf file
- **`lexical_diversity`**: Measure of the lexical diversity in the main content of .pdf file (unique_words/total_words)
- **`Department_new`**: The department of DTU from which the thesis is published
- **`grading_scientific_contribution`**: Sub grading score, (x-y)
- **`grading_methodological_rigor`**: Sub grading score, (x-y)
- **`grading_technical_implementation`**: Sub grading score, (x-y)
- **`grading_literature_review`**: Sub grading score, (x-y)
- **`grading_process_professionalism`**: Sub grading score, (x-y)
- **`grading_impact_applicability`**: Sub grading score, (x-y)
- **`grading_research_question_alignment`**: Sub grading score, (x-y)
- **`grading_total_score`**: Total assigned grading score (1-100) for the thesis by local LLM. Consistes of the scores; scientific contribution, methodological rigor, technical implementation, literature review, process professionalism, impact applicability.
- **`num_authors`**: Number of authors for MSc Thesis, count of semicolons inn column `BY`. If the value is missing (NaN), fillna(0) treats it as 0 semicolons, resulting in 1 author.
- **`handin_month_num`** getting only the month from column `handin_month` and mapping to a number (1-12) using the calendar module for robustness.


**COLUMNS IN file, BUT NOT RELEVANT FOR ANALYSIS (or is a dublication)**
- Timestamp
- Author
- ID
- member_id_ss
- primary_member_id_s
- Title
- pdf_file
- num_words_full
- num_words_cont
- pdf_sha256
- grading_meta_attempts
- grading_meta_original_chars
- grading_meta_trimmed_at_references
- grading_meta_input_chars
- grading_meta_estimated_input_tokens
- grading_meta_was_truncated
- grading_meta_prompt_fit_attempts
- grading_meta_timeout_seconds
- grading_meta_stream_mode
