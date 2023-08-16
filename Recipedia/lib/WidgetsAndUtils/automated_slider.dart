/*
import 'dart:async';
import 'package:flutter/material.dart';

class AutomatedSlider extends StatefulWidget {
  final List<Widget> items;
  final int itemCount;

  const AutomatedSlider({
    super.key,
    required this.items,
    required this.itemCount,
  });

  @override
  _AutomatedSliderState createState() => _AutomatedSliderState();
}

class _AutomatedSliderState extends State<AutomatedSlider> {
  late PageController _pageController;

  @override
  void initState() {
    super.initState();
    _pageController = PageController(
      initialPage: 0,
      viewportFraction: 0.8,
    );
    WidgetsBinding.instance!.addPostFrameCallback((_) {
      // set up a timer to automate the sliding
      Timer.periodic(const Duration(seconds: 3), (Timer timer) {
        if (_pageController.page == widget.itemCount - 1) {
          _pageController.jumpToPage(0);
        } else {
          _pageController.nextPage(
            duration: const Duration(milliseconds: 500),
            curve: Curves.ease,
          );
        }
      });
    });
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Container(
      height: size.width * 0.4,
      margin: const EdgeInsets.only(
        left: 8.0,
        right: 8.0,
        top: 24.0,
        bottom: 12.0,
      ),
      child: PageView.builder(
        controller: _pageController,
        itemCount: widget.itemCount,
        itemBuilder: (BuildContext context, int index) {
          return Container(
            margin: const EdgeInsets.only(
              right: 8.0,
              left: 8.0,
            ),
            child: widget.items[index % widget.itemCount],
          );
        },
      ),
    );
  }
}
*/

import 'dart:async';
import 'package:flutter/material.dart';

class AutomatedSlider extends StatefulWidget {
  final List<Widget> items;
  final int itemCount;

  const AutomatedSlider({
    Key? key,
    required this.items,
    required this.itemCount,
  }) : super(key: key);

  @override
  _AutomatedSliderState createState() => _AutomatedSliderState();
}

class _AutomatedSliderState extends State<AutomatedSlider> {
  late final PageController _pageController = PageController(
    initialPage: 0,
    viewportFraction: 0.8,
    keepPage: false, // added to prevent reverse scrolling
  );

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // set up a timer to automate the sliding
      Timer.periodic(const Duration(seconds: 3), (Timer timer) {
        if (_pageController.page == widget.itemCount - 1) {
          _pageController.jumpToPage(0);
        } else {
          _pageController.nextPage(
            duration: const Duration(milliseconds: 500),
            curve: Curves.ease,
          );
        }
      });
    });
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return SizedBox(
      height: size.height * 0.20,
      width: size.width * 1,
      child: PageView.builder(
        controller: _pageController,
        itemCount: widget.itemCount,
        itemBuilder: (BuildContext context, int index) {
          return Container(
            margin: const EdgeInsets.only(
              right: 20.0,
            ),
            child: widget.items[index % widget.itemCount],
          );
        },
      ),
    );
  }
}
