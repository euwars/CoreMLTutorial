//
//  ViewController.swift
//  CoreMLTutorial
//
//  Created by Farzad Nazifi on 27.09.17.
//  Copyright © 2017 Farzad Nazifi. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        let wordsFile = Bundle.main.path(forResource: "testMessage", ofType: "txt")
        let wordsFileText = try! String(contentsOfFile: wordsFile!, encoding: String.Encoding.utf8).lowercased().prepared
        let tfidf = String.tfidf(message: wordsFileText)
        let res = try! MessageClassifier().prediction(message: tfidf)
        print("your message is \(res.label) with the probability: \(res.classProbability)")
    }
}

extension String {
    func removeSpecialCharsFromString() -> String {
        var text = self
        let list = "!@#$%^&*(){}0987654321۰۹۸۷۶۵۴۳۲۱:|<>?~±!❊٪؟‌\"-/._+,،'\\;"
        list.forEach { (char) in
            text = text.replacingOccurrences(of: String(char), with: " ")
        }
        return text
    }
    
    var condensedWhitespace: String {
        let components = self.components(separatedBy: .whitespaces)
        return components.filter { !$0.isEmpty }.joined(separator: " ")
    }
    
    var removeArabic: String {
        return self.replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "ي", with: "ی").replacingOccurrences(of: "ك", with: "ک").replacingOccurrences(of: "\r", with: " ").replacingOccurrences(of: "\r\n", with: " ")
    }
    
    var prepared: String {
        let x = self.unicodeScalars.map { (scalar) -> String in
            if NSCharacterSet.letters.contains(scalar) || NSCharacterSet.whitespacesAndNewlines.contains(scalar) {
                return String(scalar)
            }
            return " "
        }
        return x.joined().removeArabic.condensedWhitespace
    }
    
    static func tfidf(message: String) -> MLMultiArray {
        let wordsFile = Bundle.main.path(forResource: "Words", ofType: "txt")
        let smsFile = Bundle.main.path(forResource: "Archive", ofType: "txt")
        do {
            let wordsFileText = try String(contentsOfFile: wordsFile!, encoding: String.Encoding.utf8)
            var wordsData = wordsFileText.components(separatedBy: .newlines)
            wordsData.removeLast()
            let smsFileText = try String(contentsOfFile: smsFile!, encoding: String.Encoding.utf8)
            var smsData = smsFileText.components(separatedBy: .newlines)
            smsData.removeLast()
            let wordsInMessage = message.split(separator: " ")
            let vectorized = try MLMultiArray(shape: [NSNumber(integerLiteral: wordsData.count)], dataType: MLMultiArrayDataType.double)
            for i in 0..<wordsData.count{
                let word = wordsData[i]
                if message.contains(word){
                    var wordCount = 0
                    for substr in wordsInMessage{
                        if substr.elementsEqual(word){
                            wordCount += 1
                        }
                    }
                    let tf = Double(wordCount) / Double(wordsInMessage.count)
                    var docCount = 0
                    for sms in smsData{
                        if sms.contains(word) {
                            docCount += 1
                        }
                    }
                    let idf = log(Double(smsData.count) / Double(docCount))
                    vectorized[i] = NSNumber(value: tf * idf)
                } else {
                    vectorized[i] = 0.0
                }
            }
            return vectorized
        } catch {
            return MLMultiArray()
        }
    }
}
