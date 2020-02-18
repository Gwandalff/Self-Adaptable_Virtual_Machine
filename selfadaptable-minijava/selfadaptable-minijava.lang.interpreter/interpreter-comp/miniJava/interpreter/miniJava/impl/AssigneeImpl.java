package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.Assignee;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

public class AssigneeImpl extends MinimalEObjectImpl.Container implements Assignee {
	protected AssigneeImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.ASSIGNEE;
	}
}
