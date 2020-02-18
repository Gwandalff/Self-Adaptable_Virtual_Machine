package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import miniJava.visitor.miniJava.Block;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Statement;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;
import visitor.VisitorInterface;

public class BlockImpl extends StatementImpl implements Block {
	protected EList<Statement> statements;

	protected BlockImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.BLOCK;
	}

	public EList<Statement> getStatements() {
		if (statements == null) {
			statements = new EObjectContainmentEList<Statement>(Statement.class, this, MiniJavaPackage.BLOCK__STATEMENTS);
		}
		return statements;
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.BLOCK__STATEMENTS :
				return ((InternalEList<?>) getStatements()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.BLOCK__STATEMENTS :
				return getStatements();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.BLOCK__STATEMENTS :
				getStatements().clear();
				getStatements().addAll((Collection<? extends Statement>) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.BLOCK__STATEMENTS :
				getStatements().clear();
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.BLOCK__STATEMENTS :
				return statements != null && !statements.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Block(this);
	}
}
