import 'package:flutter/material.dart';

// ignore: must_be_immutable
class ProgressBar extends StatelessWidget {
  String message;

  ProgressBar({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Dialog(
      child: Container(
        margin: const EdgeInsets.all(15.0),
        width: double.infinity,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(6.0),
        ),
        child: Padding(
          padding: const EdgeInsets.all(15.0),
          child: Row(
            children: [
              const SizedBox(
                width: 6.0,
              ),
              const CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(Color(0XFFFF4F5A)),
              ),
              const SizedBox(
                width: 26.0,
              ),
              Text(
                message,
                style: const TextStyle(color: Colors.black, fontSize: 14.0),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
