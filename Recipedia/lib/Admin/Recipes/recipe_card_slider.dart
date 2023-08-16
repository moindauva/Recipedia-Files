import 'package:flutter/material.dart';
import 'package:recipedia/Admin/recipes/update_or_delete_recipe.dart';
import 'package:recipedia/Admin/recipes/view_recipe_details.dart';

class RecipeCardSlider extends StatelessWidget {
  final String id;
  final String title;
  final String description;
  final int rating;
  final String cookTime;
  final String thumbnailUrl;
  final String category;
  final String diet;
  final String course;
  final String ingredients;
  final bool isAdmin;

  const RecipeCardSlider({
    Key? key,
    required this.isAdmin,
    required this.id,
    required this.title,
    required this.description,
    required this.cookTime,
    required this.rating,
    required this.thumbnailUrl,
    required this.diet,
    required this.course,
    required this.ingredients,
    required this.category,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          PageRouteBuilder(
            transitionDuration: const Duration(seconds: 1),
            transitionsBuilder: (context, animation, animationTime, child) {
              animation = CurvedAnimation(
                  parent: animation, curve: Curves.fastLinearToSlowEaseIn);
              return ScaleTransition(
                scale: animation,
                alignment: Alignment.center,
                child: child,
              );
            },
            pageBuilder: (context, animation, animationTime) {
              return isAdmin
                  ? UpdateOrDeleteRecipe(
                      id: id,
                      title: title,
                      description: description,
                      cookTime: cookTime,
                      rating: rating,
                      diet: diet,
                      course: course,
                      thumbnailUrl: thumbnailUrl,
                      ingredients: ingredients,
                      category: category,
                    )
                  : ViewRecipeDetails(
                      ID: id,
                      title: title,
                      description: description,
                      cookTime: cookTime,
                      diet: diet,
                      course: course,
                      rating: rating,
                      thumbnailUrl: thumbnailUrl,
                      ingredients: ingredients,
                    );
            },
          ),
        );
      },
      child: Container(
        width: MediaQuery.of(context).size.width,
        height: 175,
        decoration: BoxDecoration(
          color: Colors.transparent,
          borderRadius: BorderRadius.circular(15),
          /*boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.6),
              offset: const Offset(
                0.0,
                10.0,
              ),
              blurRadius: 10.0,
              spreadRadius: -6.0,
            ),
          ],*/
          image: DecorationImage(
            colorFilter: ColorFilter.mode(
              Colors.black.withOpacity(0.25),
              BlendMode.multiply,
            ),
            image: NetworkImage(thumbnailUrl),
            fit: BoxFit.cover,
          ),
        ),
        child: Stack(
          children: [
            Align(
              alignment: Alignment.center,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 5.0),
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 19,
                    color: Colors.white,
                  ),
                  overflow: TextOverflow.ellipsis,
                  maxLines: 2,
                  textAlign: TextAlign.center,
                ),
              ),
            ),
            Align(
              alignment: Alignment.bottomLeft,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    padding: const EdgeInsets.all(5),
                    margin: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.4),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.star,
                          color: Color(0XFFFF4F5A),
                          size: 18,
                        ),
                        const SizedBox(width: 7),
                        Text(
                          rating.toString(),
                          style: const TextStyle(
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.all(5),
                    margin: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.4),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.schedule,
                          color: Color(0XFFFF4F5A),
                          size: 18,
                        ),
                        const SizedBox(width: 7),
                        Text(
                          cookTime,
                          style: const TextStyle(
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  )
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
