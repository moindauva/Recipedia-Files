import 'package:flutter/material.dart';

void snackBar(BuildContext context, message) {
  final snackBar = SnackBar(
    content: Text(message,
        style: const TextStyle(
            color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16.0)),
    backgroundColor: const Color(0XFFFF4F5A),
    //backgroundColor: Colors.white70,
    behavior: SnackBarBehavior.floating,
    margin: const EdgeInsets.all(15.0),
    elevation: 30,
  );
  ScaffoldMessenger.of(context).showSnackBar(snackBar);
}
