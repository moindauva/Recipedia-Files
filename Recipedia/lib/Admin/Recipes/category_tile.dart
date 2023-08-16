import 'package:flutter/material.dart';

class CategoryTile extends StatelessWidget {
  final String image;
  final String text;
  final bool selected;

  const CategoryTile({
    Key? key,
    required this.image,
    required this.text,
    required this.selected,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(left: 12.5, right: 12.5),
      width: 70,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        color: selected ? const Color(0XFFFF4F5A) : Colors.white,
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Image.asset(
            image,
            height: 30,
            width: 30,
          ),
          Text(
            text,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: selected ? Colors.white : Colors.black,
              fontSize: 12.0,
            ),
          )
        ],
      ),
    );
  }
}
