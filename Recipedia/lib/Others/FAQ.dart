import 'package:flutter/material.dart';
import '../WidgetsAndUtils/FAQTile.dart';

class FAQ extends StatelessWidget {
  FAQ({Key? key}) : super(key: key);
  List<Map<String, dynamic>> QnA = [
    {
      'question': 'What is Recipedia?',
      'answer':
          'Recipedia is a recipe search platform that allows users to discover and find recipes for various meals, such as breakfast, lunch, dinner, and dessert. The platform provides recipe information, including ingredients, steps, and cooking time, helping users to easily find and prepare their desired dishes.',
    },
    {
      'question': 'What is unique in Recipedia?',
      'answer':
          'It has a unique feature that is "scan" feature, which allows users to scan a vegetable or fruit and find recipes that incorporate it. This feature provides users with a convenient and innovative way to discover new recipes and to find inspiration for meals, based on the ingredients they have available.',
    },
    {
      'question':
          'What inspired you to create Recipedia and what problem does it solve?',
      'answer':
          'Recipedia was created to provide users with an easy and convenient way to discover and find recipes for various meals, such as breakfast, lunch, dinner, and dessert. The platform helps users easily find and prepare their desired dishes by providing recipe information, including ingredients, steps, and cooking time.',
    },
    {
      'question':
          'Who is the target audience for Recipedia and what kind of recipes are offered?',
      'answer':
          'Recipedia is designed for people who are interested in finding and preparing various meals, including breakfast, lunch, dinner, and dessert. The platform offers a wide range of recipes for these meal categories.',
    },
    {
      'question':
          'How does Recipedia help users find and prepare their desired dishes?',
      'answer':
          'Recipedia helps users find and prepare their desired dishes by providing them with recipe information, including ingredients, steps, and cooking time. This makes it easy for users to find and prepare the recipes they want.',
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'Frequently Asked Questions',
          textAlign: TextAlign.center,
        ),
      ),
      body: ListView.builder(
        scrollDirection: Axis.vertical,
        padding: const EdgeInsets.only(top: 0),
        itemCount: QnA.length,
        itemBuilder: (context, index) {
          return FAQTile(
            question: QnA[index]['question'],
            answer: QnA[index]['answer'],
          );
        },
      ),
    );
  }
}
