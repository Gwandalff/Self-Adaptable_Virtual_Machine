package org.tetrabox.minijava.xtext.validation

import com.google.inject.Inject
import org.eclipse.emf.ecore.EObject
import miniJava.AccessLevel
import miniJava.Clazz
import miniJava.Member
import org.tetrabox.minijava.xtext.typing.MiniJavaTypeConformance

import static extension org.eclipse.xtext.EcoreUtil2.*

class MiniJavaAccessibility {

	@Inject extension MiniJavaTypeConformance

	def isAccessibleFrom(Member member, EObject context) {
		val contextClass = context.getContainerOfType(Clazz)
		val memberClass = member.getContainerOfType(Clazz)
		switch (contextClass) {
			case contextClass === memberClass:
				true
			case contextClass.isSubclassOf(memberClass):
				member.access != AccessLevel.PRIVATE
			default:
				member.access == AccessLevel.PUBLIC
		}
	}
}
