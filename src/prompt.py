from langchain_core.prompts import PromptTemplate

classifier_prompt = PromptTemplate(
    input_variables=["predicted_disease", "predicted_probability"],
    template="""
You are a knowledgeable assistant in the medical field.

A CNN-based model has analyzed a medical image and predicted the following:

- **Predicted Disease**: {predicted_disease}  
- **Prediction Confidence**: {predicted_probability}

Please provide a clear, well-researched, and comprehensive explanation of this disease. Include:
- A brief description of the disease
- Common symptoms
- Causes and risk factors
- Standard treatment options
- Any necessary follow-up or precautions

Keep the explanation medically accurate yet easy to understand.
"""
)

QA_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful and knowledgeable assistant in the medical field.

The user is asking a health-related question. Provide a clear, well-researched, and comprehensive response. Be sure to:
- Explain the answer in simple, understandable terms
- Provide context (e.g., symptoms, treatments, risk factors, prevention)
- Reference medical facts or research where applicable

Question: {question}
Answer:
"""
)
