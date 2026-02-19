"""
Practice with Label Encoding
"""

from sklearn.preprocessing import LabelEncoder
import numpy as np

# Exercise 1a: Basic encoding
print("=== Exercise 1a: Basic Encoding ===")
cities = ["London", "Paris", "Tokyo", "London", "Paris"]
le = LabelEncoder()

# MUST fit first (or use fit_transform)
encoded = le.fit_transform(cities)  # ✅ This learns AND encodes

print(f"Original: {cities}")
print(f"Encoded: {encoded}")
print(f"Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# YOUR TASK: What number is 'Tokyo'? What city is number 0?
print("\n💡 Tokyo is number:", encoded[2])
print("💡 Number 0 is city:", le.classes_[0])


new_cities = ["Paris", "Tokyo", "London"]
le.transform(new_cities)  # ❌ Will fail because 'Berlin' is unseen
# Exercise 1b: Decode back
print("\n=== Exercise 1b: Decoding ===")
decoded = le.inverse_transform(encoded)
print(f"Decoded back: {decoded}")

# YOUR TASK: Decode these numbers: [1, 0, 2]
mystery_numbers = [1, 0, 2]
mystery_decoded = le.inverse_transform(mystery_numbers)
print(f"Mystery numbers {mystery_numbers} decoded: {mystery_decoded}")

# Exercise 1c: Transform NEW data (encoder already fitted)
print("\n=== Exercise 1c: Transform New Data (Encoder Already Fitted) ===")
new_cities = ["Paris", "Tokyo", "London"]
new_encoded = le.transform(new_cities)  # ✅ Now transform works!
print(f"New cities: {new_cities}")
print(f"New encoded: {new_encoded}")
print("✅ Same cities get same numbers (consistent mapping)!")

# Exercise 1d: What about unseen cities?
print("\n=== Exercise 1d: Handling Berlin (Unseen City) ===")
try:
    berlin_encoded = le.transform(["Berlin"])  # ❌ Will fail
except ValueError as e:
    print(f"❌ Error: {e}")
    print("💡 Berlin was never in the training data!")

print("\n✅ Solution: Handle unseen values")


def safe_transform(values, encoder, unknown_value=-1):
    """Transform values, use unknown_value for unseen items"""
    result = []
    for val in values:
        if val in encoder.classes_:
            result.append(encoder.transform([val])[0])
        else:
            print(f"  ⚠️  '{val}' not seen, using {unknown_value}")
            result.append(unknown_value)
    return np.array(result)


mixed_cities = ["London", "Berlin", "Paris", "Madrid", "Tokyo"]
safe_encoded = safe_transform(mixed_cities, le)
print(f"\nCities: {mixed_cities}")
print(f"Encoded: {safe_encoded}")

print("\n" + "=" * 60)
print("📚 KEY LEARNINGS:")
print("=" * 60)
print("1. fit_transform() = fit() + transform() in one step")
print("2. fit() learns the mapping (London=0, Paris=1, etc.)")
print("3. transform() uses the learned mapping")
print("4. Must fit() before transform() works")
print("5. Unseen values cause errors - handle them!")
print("=" * 60)
