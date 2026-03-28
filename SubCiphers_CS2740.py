#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:50:04 2026

@author: praneelt
"""
import matplotlib.pyplot as plt
import random
import math
import string
import re
import urllib.request

def loadText(text):
    with open(text, "r") as fin:
        text = fin.read()
        
    text = text.lower()
    text = re.sub(r'[^a-z]','', text)
    
    return text

def bigramLog(text):
    bigramZip = zip(text,text[1:])
    counts = {}
    
    for a, b in bigramZip:
        bigram = a + b
        counts[bigram] = counts.get(bigram, 0) + 1
    
    for a in string.ascii_lowercase:
        for b in string.ascii_lowercase:
            if (a + b) not in counts:
                counts[a + b] = 1
    
    total = sum(counts.values())
    
    logProb = {bigram: math.log(count / total) for bigram, count in counts.items()}
    
    return logProb

def bigramPlot(logProb, topN=20):
    sortedBigrams = (sorted(logProb.items(), key = lambda x: x[1], reverse = True))
    topBigrams = sortedBigrams[:topN]
    
    bigrams = [bigram for bigram, value in topBigrams]
    values = [value for bigram, value in topBigrams]
    
    fig, ax = plt.subplots(figsize = (12,5))
    
    ax.bar(bigrams, values)
    ax.set_xlabel("bigram")
    ax.set_ylabel("log probability")
    ax.set_title(f"Top {topN} most common bigrams in sample text")

    plt.show()
    
def scoreText(text, logProb, minScore):
    bigrams = zip(text, text[1:])
    return sum(logProb.get(a + b, minScore) for a, b in bigrams)

def applyKey(cipherText, key):
    return ''.join(key.get(char, char) for char in cipherText)

def proposeNewKey(key):
    newKey = key.copy()
    a, b = random.sample(list(newKey.keys()), 2)
    newKey[a], newKey[b] = newKey[b], newKey[a]
    return newKey

def metropolisHastingsStep(cipherText, currentKey, currentScore, logProb, minScore):
    proposedKey = proposeNewKey(currentKey)
    proposedScore = scoreText(applyKey(cipherText, proposedKey), logProb, minScore)
    
    logAcceptance = proposedScore - currentScore
    acceptance = min(1, math.exp(min(logAcceptance, 0)))
    
    if random.random() < acceptance:
        return proposedKey, proposedScore
    else:
        return currentKey, currentScore
    
def runMCMC(cipherText, logProb, minScore, nIterations, originalText=None, printEvery=1000):
    letters = list(string.ascii_lowercase)
    shuffled = letters.copy()
    random.shuffle(shuffled)
    currentKey = dict(zip(letters, shuffled))
    
    currentScore = scoreText(applyKey(cipherText, currentKey), logProb, minScore)
    bestKey = currentKey
    bestScore = currentScore
    
    for i in range(nIterations):
        currentKey, currentScore = metropolisHastingsStep(cipherText, currentKey, currentScore, logProb, minScore)
        if currentScore > bestScore:
            bestScore = currentScore
            bestKey = currentKey
        
        if (i + 1) % printEvery == 0:
            snapshot = applyKey(cipherText, bestKey)[:500]
            
            if originalText is not None:
                decrypted = applyKey(cipherText, bestKey)
                accuracy = sum(a == b for a, b in zip(decrypted, originalText)) / len(originalText)
                print(f"iter {i+1:5d} acc: {accuracy:.1%} {snapshot}")
            else:
                print(f"iter {i+1:5d} {snapshot}")
    
    return bestKey

def runWithRestarts(cipherText, logProb, minScore, nIterations, nRestarts, originalText=None):
    bestKey = None
    bestScore = float('-inf')
    
    for i in range(nRestarts):
        key = runMCMC(cipherText, logProb, minScore, nIterations, originalText=originalText, printEvery=1000)
        score = scoreText(applyKey(cipherText, key), logProb, minScore)
        if score > bestScore:
            bestScore = score
            bestKey = key
    
    return bestKey

def main():
    
    # Reference Text
    url = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
    urllib.request.urlretrieve(url, "frankenstein.txt")

    log_probs = bigramLog(loadText("frankenstein.txt"))

    # Visualize top bigrams
    bigramPlot(log_probs, 20)

    # Creating Message
    original_message = loadText("beeMovie.txt")[500:1500]
    original_message = re.sub(r'[^a-z]', '', original_message.lower())

    # Cipher Key Generation
    letters = list(string.ascii_lowercase)
    shuffled = letters.copy()
    random.shuffle(shuffled)
    encryption_key = dict(zip(letters, shuffled))

    # Ciphertext
    ciphertext = applyKey(original_message, encryption_key)
    print(f"Encrypted Message: {ciphertext[:500]}")

    # MCMC Solver
    minScore = min(log_probs.values())

    best_decryption_key = runWithRestarts(
        ciphertext,
        log_probs,
        minScore,
        nIterations=5000,
        nRestarts=5,
        originalText=original_message
    )

    # See the result
    decrypted_message = applyKey(ciphertext, best_decryption_key)
    print(f"\nDecrypted Message: {decrypted_message[:500]}")

    true_decryption_key = {v: k for k, v in encryption_key.items()}
    print("\nOriginal Message: " + applyKey(ciphertext, true_decryption_key)[:500])


if __name__ == "__main__":
    main()