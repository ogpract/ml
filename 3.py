# Aim: To Implement Logistic Regression 


from math import exp

# Input data
x1 = list(map(float, input("Enter Annual Income (Lakhs): ").split())) 
x2 = list(map(float, input("Enter Savings (Lakhs): ").split())) 
y = list(map(int, input("Enter Loan Sanction (0 or 1): ").split())) 

# Check if inputs are valid
if len(x1) != len(x2) or len(x1) != len(y):
    print("Error: Mismatched input lengths.")
    exit()

# Initialize coefficients 
b0, b1, b2 = 0, 0, 0 

# Learning rate 
alpha = 0.3 

# Training loop 
for i in range(len(x1)): 
    # Sigmoid function
    prediction = 1 / (1 + exp(-(b0 + b1 * x1[i] + b2 * x2[i]))) 
    
    # Error
    error = y[i] - prediction 
    
    # Gradient descent updates
    b0 += alpha * error * prediction * (1 - prediction) 
    b1 += alpha * error * prediction * (1 - prediction) * x1[i] 
    b2 += alpha * error * prediction * (1 - prediction) * x2[i] 
    
    print(f"BO: {b0}, B1: {b1}, B2: {b2}") 

# Test input 
testx1 = float(input("Enter Annual Income (Lakhs) for testing: ")) 
testx2 = float(input("Enter Savings (Lakhs) for testing: ")) 

# Make a prediction
test_prediction = 1 / (1 + exp(-(b0 + b1 * testx1 + b2 * testx2)))

# Output result
if test_prediction > 0.5: 
    print("Loan Sanctioned") 
else: 
    print("Loan Not Sanctioned")
