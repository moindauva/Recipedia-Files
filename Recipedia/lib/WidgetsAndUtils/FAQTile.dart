import 'package:flutter/material.dart';

class FAQTile extends StatefulWidget {
  final String question;
  final String answer;

  const FAQTile({
    Key? key,
    required this.question,
    required this.answer,
  }) : super(key: key);

  @override
  State<FAQTile> createState() => _FAQTileState();
}

class _FAQTileState extends State<FAQTile> {
  bool isExpanded = false;

  @override
  Widget build(BuildContext context) {
    return ExpansionTile(
      title: Text(
        widget.question,
        style: const TextStyle(
          fontWeight: FontWeight.bold,
        ),
      ),
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(widget.answer),
        ),
      ],
    );
  }
}
